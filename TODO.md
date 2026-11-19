Improvements (mnist, Release, RX 6900 XT: HEAD 12.7 ms -> 7.5 ms per batch, PyTorch/ROCm 9.5 ms):

GPU kernels
4. Several outputs per work item (register blocking). Largest remaining gain: by hand
 conv2 forward 0.56 -> 0.30 ms (4 output channels), weight gradient 1.8 -> 0.35 ms
 (8 outputs, chunks), input gradient not explored. Needs accumulator arrays, an inner
 loop over the outputs, a smaller launch range.
5. Layout copies: only the dimension of the first launch id is moved last. Sort all
 dimensions by atom speed (innermost launch id fastest, then loops inner to outer);
 `IndexCode::flatten` folds only trailing blocks. Forward and input gradient want
 opposite weight layouts (0.8 ms vs 2.4 ms), so the node's own data stays untouched.
6. Loop interchange: the innermost loop should walk the smallest stride.
7. Local memory tiling: a work group staging a load that does not depend on a launch
 id. Measured on mnist with a hand-written kernel: no gain. Revisit with a heavier
 workload (matmul-shaped, attention). Needs explicit local sizes (`oclimpl.cpp`, TODO
 there): group size a multiple of the preferred multiple, pad the global range, guard
 every id separately, predicate instead of early return once barriers exist, local
 size in the kernel cache key if tile sizes are constants.
8. `MIN_REDUCTION_PARALLELISM` (`graph.cpp`, 1 << 20): a GPU is saturated at ~70k work
 items, lowering it made no measurable difference since the loops are nested. Retune
 on a second workload.
9. Unexplained: why the write index from coordinates is faster (weight gradient 1.8 ->
 1.3 ms in the harness), why the weight gradient is 1.8 ms in the app but 1.3 ms by
 hand.
10. CPU side of a batch: code generation ~0.1 ms per execution in Debug, cache lookup
 ~0.2 ms. A structural hash of the graph instead of the full source as cache key.

Scheduler (`execute_subgraphs` in `graph.cpp`)
11. `fCalculateGradients` still executes adjoints at construction time
 (`gradients.cpp:184`, `worth_materializing` in `utils.hpp`). The gradient-context
 signal in `estimate_evals` is a stopgap for it.
12. `fExecuteGraphs` for several roots (loss and all new weights of the trainer in one
 call instead of `materialize_graph` per node), the scheduler on the union of the
 graphs.
13. Multi-output kernels: side results exist for nodes read at the root's index; nodes
 in another iteration space (below a reduction) still get a kernel of their own.
14. Constants (`MATERIALIZE_COST`, `LAUNCH_COST`, `WORK_ITEM_LIMIT`, `MIN_WORK_ITEMS`)
 are tuned on mnist only.
15. Max/min/mul reductions are not split over the axes a flatten merged (every level
 needs its own gradient mask, it was slower). `findex_set` still executes indices and
 values at construction time.

Index representation (`mutate_index`)
16. Every operation maps its parameters through `mutate_index` now (inverse broadcasting
 included, aligned at the front); the flat path and its helpers (`index_defs`,
 `index_bound`, `index_mod`, ...) are gone. `IndexImpl` and `SetIndexImpl` push their
 parameters themselves (`OCL_LAZY_DONT_PUSH_PREDS`), they need a runtime value as an atom.
17. CSE still decides with `same_index` / `passesIndexOn` / `collectReusable`: a node is
 shared only on paths that pass the root's index on. With maps it becomes an equality
 of coordinates, which also works inside loops (needs scoping: a definition has to
 dominate all its uses).

Features
18. Deep learning layers for denoising / vision (NAFNet as first target): depthwise
 (grouped) convolution (`fconvolve` mixes all channels), LayerNorm layer (compose from
 reduce, `sqrt`; ONNX mapping), PixelShuffle (reshape + transpose), L1 loss, AdamW,
 LR schedule, gradient clipping, weight EMA. Data: RAW / 16-bit input (preprocess
 outside, custom `DataLoader`), noise synthesis on the GPU, tiled inference with
 overlap. Use it as the second benchmark next to mnist.
19. Data types FP16, INT16, INT8 (FP8 only if there is hardware for it). Suggestion: as
 storage types first (convert on load and store in the kernel and on the CPU, compute
 in FLOAT32 / INT32): dtype enum, codegen load/store, CPU conversion, serialization,
 image and ONNX loaders. Real FP16 arithmetic needs `cl_khr_fp16`; INT8 inference needs
 scales and int32 accumulation, which is more than a dtype.
20. Faster CPU backend with x86 code generation (asmjit, vector instructions). The CPU
 backend executes node by node without fusion, so it is memory bound; it only pays
 off if it uses the same fused graph and index maps as the GPU codegen, i.e. a second
 emitter. Measure the OpenCL backend on pocl first.
