# Execution-Time Scheduling

This document describes how Flint should decide which nodes are computed in a
kernel of their own ("early execution", materialization) and which are fused
lazily into their consumers. These decisions are made when `fExecuteGraph`
runs (`execute_subgraphs` in `src/graph.cpp`), where the whole graph that is
about to be executed is known, and no longer while the graph is built.
Sections 1–4 and 12 describe what is implemented, sections 7 and 11 partly
what is still open.

## 1. Current state

Nodes are executed early at these places:

| Where | Rule |
|---|---|
| `addNode`, `freshape`, `fslice`, `fextend` | **removed**: the 4th reference executed the predecessor if `worth_materializing` held |
| `reduce_operation` | **removed**: nested reductions were executed unless the outer one read the inner one only once |
| `materialize_permutations` | **removed**: a transpose in front of a reduction was executed if it was small enough |
| `findex_set` | indices and values are always executed |
| `fCalculateGradients` (`src/gradients.cpp:184`) | an adjoint with more than one reader or below a reduction is executed if `worth_materializing` holds |
| `funslide_window` | **removed**: unslide is lazy now and has an index map |
| `fExecuteGraph` | **new**: `execute_subgraphs` (sections 3, 4, 9) |

Problems of the construction-time rules (the reason for this plan):

1. **The decision is made too early.** The reference counter at construction
   time is a lower bound; consumers that are added later are not seen, and
   consumers that are never executed are counted. A node that was executed
   can not be fused again.
2. **The cost model only compares sizes.** It does not know how often a
   consumer evaluates each element (a repetition in front of a 288 long
   reduction evaluates every element 288 times — the GPU hang), nor how
   expensive the node is (a small node that contains a reduction loop is
   expensive to recompute).
3. **One kernel = one result.** Materializing a shared node always costs a
   launch of its own plus writing and reading it back. Per MNIST batch there
   are 54 kernels (15.8 ms kernel time), many of them below 0.1 ms, which is
   mostly launch overhead.
4. **Graphs are executed one root at a time.** The trainer executes the loss,
   then `m`, `v` and the new weight per weight (`materialize_graph` in
   `src/dl/trainer/optimizer.cpp`). Everything these graphs share is either
   materialized by the construction-time rules or recomputed per root.

## 2. Goals

- Decide materialization once per execution, on the complete graph.
- Materialize exactly where recomputation costs more than writing the node
  once and reading it back.
- Never generate a kernel whose work per work item explodes (GPU hangs).
- Keep construction-time execution only where the code generation can not do
  without it (`findex_set`).
- Prepare multi-output execution and multi-output kernels, which make
  materialization nearly free.

## 3. Evaluation count

The central quantity is how often each element of a node is evaluated when
the graph is executed. It follows from the shapes alone, no `IndexMap` is
needed.

### 3.1 Factor of an edge

For a node `N` and its predecessor `P`:

```
factor(N -> P) = elements(N) * loop_length(N) / elements(P)
```

`loop_length(N)` is the number of times one element of `N` reads `P`:

| Operation | loop_length | resulting factor |
|---|---|---|
| elementwise, reshape, transpose, convert, ... | 1 | 1 |
| repeat / broadcasting / inverse broadcasting | 1 | > 1 (the repetition) |
| sliding window | 1 | ≈ overlap of the windows (`prod(size / step)`) |
| reduction | length of the reduced axis | 1 |
| unslide | candidate windows `prod(ceil(size / step))` | ≈ 1 (each element of `P` is read once, the rest is guarded away) |
| slice | 1 | < 1 (only a part is read) |
| extend | 1 | < 1 per element of `N`, ≤ 1 overall |
| index (gather) | 1 | `elements(N) / elements(P)`, an estimate |
| concat | 1 | 1 per branch (each element of `N` reads only one parameter) |

For unslide the guard waste is better accounted for with factor 1 than with
the candidate count, since the predecessor is only evaluated behind the guard.

### 3.2 Propagation

Walk the graph from the root in topological order (every consumer before its
predecessors):

```
evals[root] = 1
for N in topological order:
    e = materialized(N) ? 1 : evals[N]
    for P in predecessors(N):
        evals[P] += e * factor(N -> P)
```

A node that is materialized is computed once in its own kernel, so it passes
on 1 instead of its own count — materializing resets the multiplication for
everything below it.

Since the decision for `N` depends on `evals[N]`, which is complete once all
consumers were visited, the decision is made when `N` is reached in the
topological order — before its predecessors are visited.

### 3.3 Examples

- **Repetition between reductions (the hang):** `reduce_sum(repeat(reduce_sum(x)))`
  with a repetition of 288: the inner reduction has `evals = 288`, it contains
  a loop → materialize. Without it every work item runs 288 inner loops.
- **Matrix multiplication** `reduce_sum(repeat(A) * repeat(B))`: `A` and `B`
  get `evals = n` resp. `m`, but if they are stores, recomputing is a load →
  keep lazy (see cost model).
- **Convolution backward** (`unslide(windows)`): factor 1 → stays lazy and fuses
  into one kernel (measured: 2.01 ms in two kernels → 1.12 ms fused).
- **Slice of an expensive node:** `evals < 1` → materializing would compute
  elements that are never read → keep lazy.

## 4. Cost model

`evals > 1` alone does not justify materializing. Recomputing a load costs
almost nothing (caches), recomputing a reduction costs its whole loop.

### 4.1 Cost per element

Implemented in `estimate_evals` (`src/graph.cpp`), bottom-up over the
topological order:

```
cost(stored node or generator) = 0      // the load is paid either way
cost(N) = own(N) + sum over predecessors P of cost(P)
cost(N) *= elements(P) / elements(N)    // only for reductions and unslide
```

`own(N)` is 0 for operations that only remap the index (reshape, flatten,
repeat, transpose, slice, extend, concat, sliding window) and 1 otherwise. A
loop executes everything below it once per element it reads, which is the
multiplication in the last line.

### 4.2 Decision

Among all nodes below the root, the one with the largest saving is executed
first:

```
saving(N) = (evals[N] - 1) * cost(N) * elements(N)
            - MATERIALIZE_COST * elements(N) - LAUNCH_COST
```

if that saving is positive. Then everything is estimated again, until no node
has a positive saving. Taking the largest saving instead of the first node in
the order matters: a node that expands what it reads (the product inside of a
convolution has a billion elements) inherits its cost from the small node
below it, which saves the same operations for a fraction of the memory.

Tuned on `examples/mnist`: `MATERIALIZE_COST = 128` (flat between 32 and 256,
5 ms slower at 8 and at 1024), `LAUNCH_COST = 1 << 16` (flat up to that, slower
above 1M).

### 4.3 Work item limit

If no node has a positive saving but `cost(root)` exceeds `WORK_ITEM_LIMIT`
(`1 << 14`, flat between 1k and 64k, 3x slower at 1M) and the root has fewer
than `MIN_WORK_ITEMS` (`1 << 14`) elements, the first loop below the root that
has at most half of the root's cost is executed in a kernel of its own (with
enough work items long ones keep the GPU busy anyway). That keeps chunked reductions (`reduce_parallel`) parallel and protects
against kernels that hang the GPU.

## 5. Uses that are free: common subexpressions

The code generation already computes a node only once per kernel if all its
uses read it at the same index (`collectReusable` in
`src/backend_ocl/codegen_lazy.cpp`: the path from the root only passes
through operations with `passesIndexOn`, and the node is no kernel
parameter). `x * x`, or two consumers of the same shape that pass the index
on unchanged, therefore cost one evaluation, not two.

The propagation counts such uses once (`shared` in `NodeEstimate`): when several consumers of `P`
reach it at the same index (same rules as `collectReusable`), `evals[P]`
is increased by the maximum of their contributions, not by the sum.
Otherwise nodes are materialized that the kernel shares for free.

## 6. Nodes used outside of the executed graph

### 6.1 The other consumers already exist

`reference_counter` counts all consumers, also those outside of the graph
that is executed. If

```
reference_counter(N) > uses of N inside the graph
```

`N` will be needed again by a later execution. Each outside use counts as
one more evaluation (`evals[N] += outside uses`). If the decision then says
materialize, `N` is executed, its `result_data` stays, and the later graph
reads it like a store.

Example: Adam reads the gradient in `new_m_expr` and in `new_v_expr`, which
are executed one after another. When `new_m` is executed, `new_v_expr`
already exists, so the gradient has an outside use → materialized once
instead of recomputed (including the whole backward pass) in the second
execution.

### 6.2 The other consumers do not exist yet

A consumer that is constructed after the execution can not be seen. If the
forward pass were executed before the backward graph is built, every
intermediate the backward pass reads would be recomputed inside of the
backward kernels.

The trainer currently builds the loss, calls `fCalculateGradients`, then
executes the loss, then per weight `m`, `v` and the new weight. Until
`fCalculateGradients` stops executing adjoints at construction time, that
order hides the problem. Afterwards, the clean solution is section 7:
execute everything that is needed in one call, then every consumer is
known (6.1).

Until then `estimate_evals` uses the gradient context as a signal: a node
that depends on a watched variable (`gradient_data` is set) and has no other
reader yet counts as read once more, since the backward pass will read it.
Without it MNIST takes 30 ms instead of 19 ms per batch (debug build), because
`fCalculateGradients` executes the forward pass before the backward nodes of
the convolutions exist.

## 7. Multi-root execution and multi-output kernels

Not implemented, except for side results (7.2) in their simple form.

### 7.1 Multi-root execution

A new entry point executes several roots in one go:

```c
FErrorType fExecuteGraphs(FGraphNode **nodes, unsigned int num_nodes);
```

The scheduler runs once on the union of the graphs (`evals` of every root is
1). Shared nodes get their `evals` from all roots, so a node that two roots
read is materialized once (or, with 7.2, written by the first kernel as a side
result).

Users:

- `fCalculateGradients` returns the gradients lazily and the caller executes
  all of them together; the per-adjoint execution in `gradients.cpp:184`
  becomes unnecessary (the scheduler sees that an adjoint has several
  readers).
- The trainer executes the loss and all new weights (`m`, `v`, weight per
  weight) in one call instead of `materialize_graph` per node.

### 7.2 Multi-output kernels

A kernel writes several results: `R0, R1, ...`. A node that is materialized
and has the same iteration space as the root of the kernel it is computed in
(same number of elements, read at the root's index — the `collectReusable`
condition) is written as a side result instead of getting a kernel of its
own:

```c
const float v7 = ...;   // the shared node
R1[index] = v7;         // side result
...
R0[index] = v0;
```

That removes the launch and the reading back; what remains is one write. The
term `C_LAUNCH` and one `C_MEM` vanish from the decision in 4.2 for these
nodes. Nodes with another iteration space (e.g. below a reduction) still get a
kernel of their own.

Required changes: the kernel signature gets one result buffer per output,
`fExecuteGraph_gpu` allocates them and sets `result_data` for every output, the
kernel cache key stays the full source.

## 8. What stays at construction time

- `findex_set`: a scatter, its indices and values must be materialized.
- `materialize_permutations` and the nested reduction rule in
  `reduce_operation` are removed, MNIST is as fast without them.
- The `worth_materializing` calls in `graph.cpp` are replaced by the
  scheduler, the one in `gradients.cpp` is still there.

## 9. The scheduler

The plan was one pass that marks every node at once:

```
schedule(roots):
    graph   = all nodes reachable from roots that have no result_data
    order   = topological order, consumers first
    inside  = number of uses of each node inside graph
    cost    = bottom-up cost (4.1), everything lazy
    evals   = 0, evals[root] = 1 for every root
    for N in order:
        evals[N] += max(0, reference_counter(N) - inside(N))   // 6.1
        if N is root or worth (4.2) or limit (4.3) or rule (8):
            mark N materialized, update cost of the nodes above it
        e = materialized(N) ? 1 : evals[N]
        for P in predecessors(N), grouped by "same index" (5):
            evals[P] += e * factor(N -> P)
    execute materialized nodes in reverse topological order
    (each is a kernel of its own, or a side result with 7.2)
```

Implemented is a simpler loop (`execute_subgraphs`): estimate everything
(`estimate_evals`), execute the node with the largest saving (4.2) or the one
the work item limit (4.3) asks for, repeat until there is none. It runs in
`fExecuteGraph` before the backend is called, only if the GPU backend is
active and the node has no result yet; the CPU backend executes node by node
without fusion. Factors and costs are computed in `graph.cpp` (`read_factor`,
`remaps_index`) instead of virtual methods per operation.

## 10. Validation

- `build/test`, `build/test_gradients` (note: the assertion count of
  `build/test` varies between runs because of random tensors).
- MNIST: batch time, loss after 5 batches (diverging loss went unnoticed by
  the tests before), number of kernels and kernel time per batch
  (`--debug`, sum of `execution took`).
- Small graphs for every rule: repetition between reductions, matrix
  multiplication, convolution forward/backward, slice of an expensive node,
  Adam's double use of the gradient.
- GPU safety: run new graphs first at small sizes, check the safety limit
  (4.3) triggers on the 288× repetition case before running MNIST.

## 11. Overview per file

Done: the `graph.cpp` row without `fExecuteGraphs`, side results in
`codegen_lazy.cpp` / `oclimpl.cpp` (single root). Open: everything else.

| File | Changes |
|---|---|
| `src/graph.cpp` | Remove early execution from `addNode`, `freshape`, `fslice`, `fextend`, `reduce_operation` (nested rule) and `materialize_permutations` (moves into the scheduler as a rule); keep it in `findex_set`. Call the scheduler in `fExecuteGraph`; add `fExecuteGraphs`. |
| `flint.h` / `flint.hpp` | Declare `fExecuteGraphs`; C++ wrapper for executing several tensors at once. |
| new `src/scheduler.hpp` / `.cpp` | Topological order, `factor`, `loop_length` per operation, bottom-up `cost`, `evals` propagation with the same-index grouping, outside uses, decision, safety limit; returns the nodes to materialize in execution order. |
| `src/operations/implementation.hpp` | Per operation: `loop_length(node, pred)` and `own_cost(node)` as virtual methods of `OperationImplementation` (defaults 1), overridden by reductions, unslide, and the expensive unary operations. |
| `src/operations/*.cpp` | The overrides of `loop_length` / `own_cost`. |
| `src/utils.hpp` | `worth_materializing` / `largest_input` become obsolete (the cost model covers the expansion case). |
| `src/backend_ocl/codegen_lazy.cpp` | Expose the `collectReusable` same-index rule so the scheduler can use it; for 7.2: generate side results (`R1[index] = ...`) for materialized nodes in the root's iteration space. |
| `src/backend_ocl/oclimpl.cpp` | `fExecuteGraph_gpu` for several outputs: one result buffer per output in the signature, allocation, `result_data` for each; execute the scheduler's nodes in order. |
| `src/backend_cpu/cpuimpl.cpp` | No scheduler needed; `fExecuteGraphs` executes the roots one after another. |
| `src/gradients.cpp` | Remove the adjoint execution (`readers > 1 \|\| reduction`); gradients are returned lazily and executed together by the caller. |
| `src/dl/trainer/optimizer.cpp` | Execute loss and all weight updates (`m`, `v`, weight for every weight) with one `fExecuteGraphs` call instead of `materialize_graph` per node. |
| `examples/mnist` | Benchmark: batch time, loss, kernel count and kernel time. |

## 12. Memory layout copies

Measured on the conv2 kernels of MNIST with hand-written kernels (Release
build, real sizes): arithmetic is not the limit, memory access is.

| conv2 forward | time |
|---|---|
| current (one flat loop of 288, weights `[oc][kh][kw][c]`) | 3.7 ms |
| nested loops over `kh`, `kw`, `c` (three reductions instead of flatten + reduce) | 1.7 ms |
| nested loops + weights as `[kh][kw][c][oc]` | 0.73 ms |
| the same + 4 output channels per work item | 0.30 ms |

Neighbouring work items are executed together. If they read neighbouring
memory that is one access, if their addresses are far apart (the weights:
`288 * oc + ...`, so 288 entries apart) it is one access per work item. A
local-memory tile gave nothing here, the input window is already one access.

### 12.1 Copies

A node keeps its data in its own layout, untouched. In addition it may have
copies of it with permuted dimensions (transposed nodes with a result),
stored in a table of the OpenCL backend (`FGraphNode` is part of the public C
struct) and freed together with the node. Only the code generation of a load
decides which one is read, everything else (CPU backend, syncing to the host,
the data pointer of the C API) keeps seeing the original.

The forward pass and the backward pass want different layouts of the same
weights (the input gradient takes 0.8 ms with the current layout and 2.4 ms
with the one the forward pass wants), so the layout of the node itself must
not be changed: it would be copied back and forth in every batch.

### 12.2 Decision

Made at the load in `generateCode`, the only place where the index map of the
node is known:

1. `reads = elements(root) * lengths of all loops around the load /
   elements(node)`. Only for `reads` far above 1 (the weights: 62000) a copy
   costs nothing compared to what it saves. A node that is read once per
   element never gets one.
2. Every atom has a speed: the innermost launch coordinate is the fastest,
   then the loop atoms from the innermost to the outermost loop, then the
   other launch coordinates. Each dimension is as fast as the fastest atom in
   its expression.
3. The wanted layout has the dimensions sorted by speed, the fastest one
   last. If the fastest dimension is not the last one and its stride is
   larger than a cache line (16 floats), the copy with that permutation is
   read instead (created lazily: the transposed node becomes the parameter
   of the kernel and is executed before the kernel is launched).
4. The index map of the load is permuted like the shape, the address is
   flattened from that.

An upper bound for the size of a copy keeps large weights (dense layers of a
transformer) from being doubled.

### 12.3 Order of the steps

1. Reduce over the dimensions a flatten merged instead of over the merged
   axis (`reduce_parallel` / `reduce_operation`), so the code generation
   emits nested loops and no div/mod. Without it the fold in
   `IndexCode::flatten` would have to learn blocks of split coordinates that
   are not the trailing ones, otherwise the permuted address gets its
   div/mod back.
2. The copies of this section.
3. `reduce_parallel` chunks the outermost merged dimension (the batch)
   instead of every n-th position, for the weight gradient (3.2 ms -> 1.7 ms).
4. Several outputs per work item (0.73 ms -> 0.30 ms), only pays off after 2.

### 12.4 State

Implemented (MNIST per batch, Release build: 12.7 ms -> 7.5 ms, PyTorch with
ROCm on the same GPU: 9.5 ms):

- `reduce_parallel` reduces the axes a FLATTEN merged instead of the merged
  axis, for sums (the gradient of max/min/mul needs every level in between).
  The flatten is freed if nothing else reads it, otherwise it keeps the whole
  forward pass alive.
- `fconvolve` gives the windows one axis per direction, `unbroadcast` reduces
  the innermost extra axis first: the weight gradient is three nested loops.
- Layout copies (12.1, 12.2) in `src/backend_ocl/codegen_lazy.cpp`, in the
  simple form: only the dimension of the first launch id is moved to the end
  (`layoutDimension`, `layoutCopy`, `freeLayoutCopies`). No sorting of the
  other dimensions yet.
- The flat index a kernel writes to is calculated from the coordinates and not
  from the launch ids (weight gradient 1.8 ms -> 1.3 ms, cause unknown).

Open: several outputs per work item (conv forward 0.56 ms -> 0.30 ms, weight
gradient 1.8 ms -> 0.35 ms by hand), `MIN_REDUCTION_PARALLELISM` (a GPU is
saturated at about 70k work items, not 1M).
