# Symbolic indices

## 1. What is wrong today

Every kernel has one variable `index`. Each operation that reads its parameter
at a different position than itself (transpose, reduce, repeat, slice, window,
reshape) takes that flat `index` apart into coordinates with divisions and
modulos, remaps one of them, and multiplies everything back together. The next
operation does the same thing again from scratch.

The hottest mnist kernel shows what that costs: 31 divisions and modulos per
loop iteration around a single multiply-add. Two stacked `FREPEAT`s alone take
the index apart and put it back together twice.

## 2. What we want instead

Carry the coordinates themselves and compute an address only where memory is
actually touched, which is at the loads.

Then most remapping operations stop computing anything:

- transpose permutes a vector of coordinates,
- a broadcast sets one coordinate to `0` and it disappears from the address,
- slice turns a coordinate into `coordinate * step + start`,
- a reduce adds one coordinate, which is its loop variable.

Only reshape genuinely needs a division, and even that usually cancels because
conv and pooling flatten a window and reshape it back.

## 3. How it works

One index map travels from the kernel's result down to the loads, which is
*against* the data flow. Each operation receives the coordinates it is read at
and passes on the coordinates its parameters are read at. Nothing is stored and
nothing points at anything — it is an attribute that is consumed while the
traversal walks down, like constant propagation, except the value is a
coordinate per dimension instead of a constant.

Example, `A` is `[6,4]` in memory, `t = transpose(A)` is `[4,6]`,
`r = reduce_sum(t, 1)` is `[4]`, and the kernel computes `r`:

| node | coordinates | atoms in play |
|---|---|---|
| `r`, the result | `(g)` | `g`, the launch id, `[0,4)` |
| `t`, read by the reduce | `(g, i)` | `i` added, `[0,6)` |
| `A`, read by the transpose | `(i, g)` | unchanged |
| the load | address `4i + g` | unchanged |

The transpose added no atom and no code, it swapped two entries. Two operations
collapse into one address, and because it is affine in `i` the loop can step by
4 instead of recomputing.

The same picture with two parameters, for a hypothetical fused matmul
`C[i,j] = sum_k A[i,k] * B[k,j]`: it receives `(i, j)`, creates one atom `k`,
and passes `(i, k)` to `A` and `(k, j)` to `B`. Same incoming coordinates, two
different outgoing maps, one shared new atom. It emits the accumulator and the
loop over `k`, and never an address.

## 4. Names

These names are used everywhere below and nowhere else means anything else.

| name | in code | what it is |
|---|---|---|
| atom | `IndexAtom` | a value the index arithmetic cannot look into |
| coordinate | `IndexExpr` | position along one dimension, `a + sum of c * atom` |
| index map | `IndexMap::expr_per_dim` | all coordinates of one node, one per dimension |
| the node's map | `state.index_map` | coordinates the node currently generated is read at |
| a parameter's map | `state.pred_index_maps[i]` | coordinates its parameter `i` is read at |
| flatten | `IndexMap::flatten` | coordinates -> one element offset |
| split | `IndexMap::split` | one element offset -> coordinates |
| flat path | — | the old route over the `index` variable |

There are exactly three kinds of atom:

1. a launch id, one per dimension of the kernel's result,
2. the loop variable of a reduce,
3. a value only known at runtime: a loaded index, a kernel argument like the
   slice offset, or a modulo that could not be resolved.

Everything else composes without creating atoms.

## 5. What is already there

- `src/operations/indices.hpp`: `IndexAtom`, `IndexExpr` (`a`, `exprs`, `bound`,
  `derive_bound`, `to_code`), `IndexMap` (`expr_per_dim`, `flatten`, `split`).
- `OCLLazyCodegenState`: `index_map`, `index_atoms`, `add_atom`,
  `pred_index_maps`.
- `CodegenTask::index_map`, so every queued node carries its coordinates.
- `OperationImplementation::mutate_index` with its default, and `trailing_map`
  in `indices.hpp`, which is the identity that default is built from.
- `generateCode` seeds the result's coordinates from the launch ids the same way
  `launchRange` assigns them, and declares one variable per coordinate.
- Loads flatten the map they were handed with the parameter's own shape.
- Every operation except the index transparent ones (step 7.1) takes the flat
  path: its coordinates are folded back into `index` before it runs, and its
  parameters get `split` of `index`.
- `test/test.cpp`, suite `Index Optimizations`: `split(flatten(x)) == x` and the
  negative coefficient case.

Both suites pass in this state, and each kernel's folded index is identical to
the old flat one. That is the baseline to keep green after every step below.

## 6. The hook

This part is in the tree. `OperationImplementation::mutate_index` reads
`state.index_map` and fills `state.pred_index_maps` with one map per parameter.
It returns false if it has no mapping, then the operation stays on the flat
path. Its default is the identity for the index transparent operations, built
with `trailing_map`: the parameter is aligned at the *trailing* end, the leading
coordinates are dropped, and a single element parameter becomes the coordinate
`0`. That is where forward broadcasting is handled, and it is why the loads no
longer need `index % num_entries`. `FLATTEN` and `FRESHAPE` are excluded
although `passesIndexOn` lists them: the flat index passes through them, the
coordinates do not (step 7.3).

The four places in `generateCode`:

1. `pred_index_maps` is cleared per node, next to `index_defs`.
2. `mutate_index` is called *before* `generate_ocl_lazy`, so an operation can
   create its atoms before it emits the loop that uses them. A partial fill is
   an `F_ERROR`: either all parameters are mapped or none, since
   `pred_index_maps.empty()` is the only flag the rest of the node looks at.
   Build the maps in a local vector and publish them once all of them worked.
3. An operation without a mapping gets its own flat index folded into `index`
   (`flatten` of its map with its own shape), skipped when that is literally
   `index`.
4. The parameters are pushed with `pred_index_maps[i]` when it is filled, else
   with `split` of the `index` atom as before. The atom is created once per
   node, not per parameter, and only when it is going to be used.

Two ordering rules for the fold, both found the hard way:

- **Save and restore `index` around the node.** It is one mutable variable and
  the parent still needs the value it left there: the root's `R[index]`, or a
  sibling parameter whose map references the atom named `index`. The bracket has
  to lie outside the operation's own `old_index` save, otherwise that captures
  the folded value and restores the wrong one afterwards (the `Transpose` test).
- **Queue the fold before `generate_ocl_lazy` runs.** `push_front` means later
  pushed is earlier in the code, and `IndexImpl` pushes its parameter tasks from
  inside `generate_ocl_lazy`. Queued after that call, the fold lands behind the
  loads it was meant to set up (the `Index` test). The restore is the opposite
  case: `code.prepend` before the call, so the operation's own code is prepended
  in front of it.

## 7. Steps, in this order

Each step is two edits in the operation and nothing else. Both edits always
belong together: `mutate_index` says where the parameters are read,
`generate_ocl_lazy` loses its index arithmetic but keeps its value code and its
loops. If you only do the first, the operation applies its remapping twice.

After each step: both suites green (`test_gradients` is the real net, the
gradients use far more combinations than any forward pass), then look at the
division and modulo count in the hot kernel and at ms/batch of mnist at its
default batch size of 512.

**7.1 Elementwise, comparison, conversion. Done.** No operation code at all,
this is the identity of the default `mutate_index`. It covers the bulk of every
graph. On its own it is 2% *slower* on mnist, because nothing consumes the
coordinates yet and every operation below one of these has to fold them back
into `index`. The folds disappear again with the steps below.

**7.2 Reduce** (`reducing` in `reductions.cpp`, covers all four).
The result has one dimension less than the parameter, so the map *gains* one:
insert a new coordinate at the reduced axis, consisting of a fresh atom with
range `[0, pred.shape[dim] - 1]`. `generate_ocl_lazy` keeps the accumulator,
the `for` and the combine, and drops `base`, `old_idx` and the `index = ...`
line. It finds the loop variable in `state.pred_index_maps[0]` at the reduced
axis — that coordinate *is* the atom, so no extra bookkeeping is needed.

**7.2b Inverse broadcasting.** The mirror of `trailing_map` and the first step
that removes arithmetic that was never cheap: `index / iv` divides the trailing
extents away, which as coordinates is simply *dropping the trailing ones*, so
the parameter's coordinate `i` is the node's coordinate `i`, aligned at the
front. It replaces two integer divisions plus the four `index` assignments the
`inverse_broadcasting` block emits, and the fold in front of them.

It does not fit the default's condition, because `passesIndexOn` rejects these
nodes and that predicate also drives `pred_same_index` and `pred_index_bound` —
leave it alone and give the default its own case for the binary operations with
`iv != 1`. The second edit is in `generateCode`: skip the `inverse_broadcasting`
block when maps were produced, otherwise the divisions are applied twice.

**7.3 Reshape and flatten** (`FlattenImpl`, used for `FLATTEN` and `FRESHAPE`).
`split(flatten(node's map, node shape), parameter shape)`. Its
`generate_ocl_lazy` already only emits an alias, so nothing is removed there.
This is the step where the cancellation rules earn their keep: a reshape that
undoes an earlier one leaves no division behind.

**7.4 Transpose** (`shape_modification.cpp`). The shape rule is
`result.shape[i] = parameter.shape[perm[i]]`, so the parameter's coordinate at
`perm[i]` is the node's coordinate at `i`. Delete the whole `working_index`
block from `generate_ocl_lazy`.

**7.5 Repeat** (`shape_modification.cpp`), three cases per dimension:
parameter size equal to the result's → take the coordinate over; parameter size
1 → coordinate `0`; otherwise a real modulo → add an atom for
`coordinate % parameter size` with range `[0, size - 1]` and emit that one
variable. Only the third case produces code.

**7.6 Slice** (`index_modification.cpp`). `coordinate * step[i] + start[i]`.
Keep the offset as the kernel argument it is today (`addScalar`), as an atom
with factor 1 — writing it as a literal would make the kernel cache miss on
every batch again.

**7.7 Sliding window** (`sliding_windows.cpp`). The result's first dimension
enumerates all window positions, so split that coordinate by the per dimension
window counts, then each parameter coordinate is
`window position * step + offset inside the window`.

**7.8 Conditions, then extend and concat.** These two need a predicate, so add
`std::vector<std::string> conditions` to `IndexMap` first and let the load emit
`condition ? load : 0`. Extend is the inverse of slice plus a range check,
concat is two maps selected by a comparison. Do not turn a failing condition
into an early `return`: once work groups and barriers arrive, every work item
has to reach the barrier.

**7.9 Leave on the flat path for now.** `IndexImpl` and `SetIndexImpl` push
their parameters themselves, `UnslideWindowImpl` has a loop with a runtime trip
count, and the inverse broadcasting path in `generateCode` inserts its own index
manipulation. They keep working unchanged because an empty map means flat path.

## 8. After the operations are migrated

- **The bound at a load.** The load currently decides about its modulo with
  `state.index_bound`, not with the coordinate's own bound, because `split`
  clamps coordinates to their dimension and that is too tight for an opaque flat
  index. Once a chain is fully symbolic the coordinates are real coordinates and
  the derived bound becomes the better source. Switch it then, not before.
- **Hoisting.** Split each address by the scope of its atoms, emit the part
  without loop variables before the loop, and let the rest step by its stride.
- **CSE.** `collectReusable` currently relies on `passesIndexOn` to decide
  whether two uses see the same index. With maps it becomes an equality of
  coordinates, which also works inside loops.
- **Cleanup.** `index_defs`, `index_bound`, `pred_index_bound`, `same_index` and
  the string helpers `index_mod`, `index_div`, `index_coordinate` all disappear
  once no operation takes the flat path.
- **Only then tiling.** Loop order and tile sizes are not expressible in the
  maps, they need the loop nest as its own data. That is a separate design and
  it is not needed for anything above.
