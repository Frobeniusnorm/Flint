Improvements:
Symbolic indices and multi-dimensional launch:
1. Index representation: keep one coordinate expression per dimension instead of a
 flat index. Expressions are affine in atoms (global ids, loop variables, loaded
 index values) with known ranges. Per operation a `mutate_index(...)` coordinate map
 instead of emitted div/mod: elementwise identity, broadcast -> 0, transpose =
 permutation, slice `c*step+start`, reduce adds a loop variable, sliding window adds
 the window offset, extend adds a predicate. Only loads linearize.
 Fallback for unmigrated ops: linearize, run the old code, continue -> migrate one op
 at a time (elementwise, reshape, repeat, transpose, reduce, sliding window, slice,
 concat, index).
2. Optimizations on these expressions:
 `(a*s+b)/s -> a` and `(a*s+b)%s -> b` for `0<=b<s`, `x%s -> x` for `x<s`, constant
 folding. Reshape stays symbolic, div/mod only where a split can't be proven.
 Also makes CSE work inside loops: same coordinates = same value.
3. Launch is independent of 1./2.: stay flat (1D) until a kernel is tiled, the 3D
 range is ~1% slower without tiling. `launchRange` only folds neighbouring axes and
 keeps id 0 on the innermost axis (coalescing).
4. Hoisting: addresses affine in a loop variable become `A + i*B`, hoist `A`, step by
 `B`. Interchange loops so the innermost one walks the smallest stride.
5. Tiling: a load whose address does not depend on a launch id is shared by the
 work group along that id -> stage it in local memory. First the contraction
 `reduce_sum(mul(broadcast, broadcast))` (conv weight gradients, matmul), then
 sliding window tiles with halo. Needs explicit local sizes:
 - group size (product over all dims) multiple of the preferred multiple, at most
   CL_KERNEL_WORK_GROUP_SIZE, per dim at most CL_DEVICE_MAX_WORK_ITEM_SIZES
 - pad the global range, guard every id separately (not the flat index), predicate
   instead of early return once barriers exist
 - local size belongs into the kernel cache key when tiles are compile time constants
6. Register blocking (each work item a small output block), then tune tile sizes.
 Afterwards revisit the graph level split reduction and `reduce_operation`'s eager
 barrier, the split currently depends on that barrier.
