# Implementing a new Operation

The new operation framework allows easy and modular extension of the current functionality of Flint.
An operation is a class that is derived from the `OperationImplementation` class in `implementations.hpp`.
After the class is implemented it has to be registered in the array in `implementations.cpp`.
Nodes that should use this implementations have to have the index in the array as their value in the `op_type` field in `FOperation`,
for official functions it is advised to extend `flint.h`  and `graph.cpp` with a node-building function that includes the type and an entry in the `FOperationType` enum in `flint.h`.

The following functions have to be overwritten:
- `FGraphNode *local_gradient(FGraphNode *y, int dx_i, FGraphNode *prev_adj)` calculates the gradient of the node `y` to its parameter `dx_i` with the adjoint `prev_adj`, see documentation
- `void execute_cpu(const FGraphNode *node, std::vector<CPUResultData> predecessor_data, void *__restrict__ result, size_t from, size_t size)` calculates the result of `node` for the indices `from` (inclusive) to `end` (exclusive) in the array `result`.
  Because working with `void*` arrays is cumbersome, several code-patterns that dispatch the calculations to other implementable but templated functions exist, see `ZEROARY_EXECUTE_IMPL`, `BINARY_EXECUTE_IMPL`, `UNARY_EXECUTE_IMPL` etc.
- `int generate_ocl_lazy(const FGraphNode *node, std::string name, OCLLazyCodegenState &compiler_state)` generates the OpenCL code for lazy execution. Only the calculation of the varialbe `name` which should hold the result for the entry `index` (predefined) is to be generated in the Twine `compiler_state.code`.
  The parent nodes are the variables `"v" + to_string(compiler_state.variable_index + 1)`,  `"v" + to_string(compiler_state.variable_index + 2)`, etc. See existing implementations and documentation.
- `std::string generate_ocl_eager(FType res_type, std::vector<FType> parameter_types)` generates the OpenCL code for the eager kernel. The index definition and kernel definition is already generated, the result has to be written in the array `R` at the index `index`.

Optional functions that may be overwritten:
- `std::string generate_ocl_parameters_eager(FType res_type, std::vector<FType> parameter_types)` generated parameters for the OpenCL eager kernel. The result type and parameter types are given as arguments. Already generated parameters include the result array and its number of entries `R` and `num_entriesR`.
- `void push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel, cl_context context, int &par_index, std::list<cl_mem> &to_free)` pushed additional parameters that have been declared in `generate_ocl_parameters_eager` to the eager OpenCL kernel.
- `void push_parameter_kernel_parameters(FGraphNode *node, FGraphNode *pred, cl_kernel kernel,cl_context context, int &par_index, std::list<cl_mem> &to_free)` does the same as `push_additional_kernel_parameters` but for each parameter (e.g. if you want to push the shape of each parameter node to the kernel this would be a more fitting place to implement it then `push_additional_kernel_parameters`, since this function is called once per parameter).
- `std::vector<std::vector<FType>> kernel_type_combinations(const FGraphNode *node)` generates the combination of parameter and return types possible. The default implementation allows all combination of parametr values and sets the return value to the highest type (overwriting this makes sense for e.g. `findex` since it only receives integer indicies, so generating double and float kernels for the indices parameter is inefficient and in the worst case leads to compilation errors).
- `int operation_score(FGraphNode *node)` return an integer that helps decide which backend to use. Helpful for tweaking the overhead of an operation.
- `void free_additional_data(FGraphNode *node)` if the operation allocates dynamic memory in `additional_data`, this function is called to free that memory upon destruction of the node.  
