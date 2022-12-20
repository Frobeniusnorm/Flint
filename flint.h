/* Copyright 2022 David Schwarzbeck

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FLINT_H
#define FLINT_H
#define CL_TARGET_OPENCL_VERSION 200
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#ifdef __cplusplus
extern "C" {
#endif
/*
  This is the basic header file and implementation of Flint, to be compatible
  with as many languages as possible. It is C compatible.

  Flints Execution structure represents an AST, so that each Graph may be
  compiled to a specific OpenCL program
*/

/** Initializes the cpu and the gpu backend. they are already implicitly called
 * by the execution functions if necessary. the first method allows disabling of
 * the gpu backend (by passing 0 to its gpu parameter), the cpu backend cannot
 * be disabled (if a 0 is passed to the cpu backend in the first method, it may
 * still be started by :func:`fExecuteGraph`, if it decides that the cpu backend
 * should be chosen). only use those functions if you...
 * - ...want to explicitly decide where and when the initialization should take
 *      place
 * - ...want to only start one backend */
void flintInit(int cpu, int gpu);
void flintInit_cpu();
void flintInit_gpu();

/**Deallocates any resourced allocated by the corresponding backends.
The first method calls the other two which are only executed if the framework
was initialized, else they do nothing. */
void flintCleanup();
void flintCleanup_cpu();
void flintCleanup_gpu();
/** Sets the logging level of the framework. Adjust this for debugging purposes,
 * or if you release software in which Flint is contained.
 * .. seealso::
 *   :func:flog, :enum:FLogType
 *
 * Levels:
 * - 0: No logging
 * - 1: Only :var:`F_ERROR`
 * - 2: Logging level 1 + :var:`F_WARNING` (should be used for production)
 * - 3: Logging level 2 + :var:`F_INFO` (for developement)
 * - 4: Logging level 3 + :var:`F_VERBOSE` (for library developement)
 * - 5: Logging level 4 + :var:`F_DEBUG` (when a bug in the library has been
 * found)*/
void fSetLoggingLevel(int);

/** - :var:`F_DEBUG` (only internal debugging informations of the framework),
 * - :var:`F_VERBOSE` (verbose information, may be helpful to users of the
 * library),
 * - :var:`F_INFO` (informational data of the framework, e.g. which
 * graphics card has been chosen),
 * - :var:`F_ERROR` (unrecoverable errors,
 * generated by function calls to the framework, raises a exception everytime),
 * - :var:`F_WARNING` (probably unwanted behaviour or undefined behaviour caused
 * by missuse of functions).*/
enum FLogType { F_DEBUG, F_VERBOSE, F_INFO, F_ERROR, F_WARNING };

/** Logs a :var:`NULL` terminated string with the given logging level. */
void flog(FLogType type, const char *msg);
// TODO the following three methods are neaded for gradient calculation (i
// think)
/** All graph nodes that represent actual operations are after this call
 * executed eagerly, i.e. they are executed during graph construction. */
void enable_eager_execution();
/** Disable eager execution, i.e. the graph is constructed without execution of
 * the nodes until a operation makes the execution of a parent graph necessary
 * or the user calls :func:`fExecuteGraph`. */
void disable_eager_execution();
/** Returns 1 if eager execution has been enabled, else 0 */
int is_eager_execution();
/** The 4 allowed data types: :var:`F_INT32` (integer, 32bit), :var:`F_INT64`
 * (integer, 64bit), :var:`F_FLOAT32` (floating point, 32bit), :var:`F_FLOAT64`
 * (floating point, 64bit)*/
enum FType { F_INT32, F_INT64, F_FLOAT32, F_FLOAT64 };
enum FOperationType {
  FSTORE,
  FRESULTDATA,
  FCONST,
  FADD,
  FSUB,
  FMUL,
  FDIV,
  FPOW,
  FNEG,
  FLATTEN,
  FMATMUL,
  FCONVERSION,
  FRESHAPE,
  FMIN,
  FMAX,
  FREDUCE_SUM,
  FREDUCE_MUL,
  FSLICE,
  FABS,
  FNUM_OPERATION_TYPES
};
/** Describes one operation. An operation always has a shape, described by
 * :member:`FOperation.shape` which is an array of size
 * :member:`FOperation.dimensions` with each entry denoting the size of the
 * corresponding dimension. :member:`FOperation.op_type` denotes the type of
 * operation, :member:`FOperation.data_type` the type of the underlying data,
 *  :member:`FOperation.additional_data` is operation specific.*/
struct FOperation {
  // shape of the data after execution
  int dimensions;
  size_t *shape;
  // type of operation, to enable switch cases and avoid v-table lookups
  FOperationType op_type;
  // datatype of result
  FType data_type;
  void *additional_data;
};
/** Describes one node in the Graph. Stores the corresponding operation in
 * :member:`FGraphNode.operation`, an array of predecessors (the arguments of
 * the operation) in :member:`FGraphNode.predecessors`, its size in
 * :member:`FGraphNode.num_predecessor` and the reference counter in
 * `FGraphNode.reference_counter`. Do not modify any parameter by yourself,
 * since the framework manages them, but you can read the data and structure
 * from them. The nodes are allocated by the operation functions, they and their
 * members should neither be manually created, edited or freed except by the
 * corresponding flint methods. */
struct FGraphNode {
  int num_predecessor;
  FGraphNode **predecessors;
  FOperation *operation;    // the operation represented by this graph node
  size_t reference_counter; // for garbage collection in free graph
};
/** Result of an call to :func:`fCreateGraph`, see :struct:`FResultData`.
 * Data of this Operation may always be changed, since the framework assumes
 * this. */
struct FStore {
  // link to gpu data
  cl_mem mem_id = nullptr;
  void *data;
  size_t num_entries;
};
// TODO dirty bit to may keep store data

/** Stores the resulting data after an execution of :func:`fExecuteGraph`.
 *  The data can be found in :c:member:`FResultData.data`, the datatype in
 * :c:member:`FOperation.data_type` of the corresponding :struct:`FGraphNode`.
 *  The number of entries (**not** number of bytes) is stored in
 * :member:`FResultData.num_entries`. The data may be consistently modified
 * if...
 *  - ...when the data size is changed, num_entries is equivalently updated and
 *        c:func:`realloc` is used
 *  - ...the data was not already loaded to the gpu (i.e. the result must be the
 *        return value of :func:`fExecuteGraph_cpu`) */
struct FResultData {
  // link to gpu data
  cl_mem mem_id = nullptr;
  void *data;
  size_t num_entries;
};
// A single-value constant (does not have to be loaded as a tensor)
struct FConst {
  void *value; // has to be one of Type
};
// range instructions
struct FSlice {
  long *start;
  long *end;
  long *step;
};
// functions

/** :param data: pointer to the flattened data array that should be loaded into
 * the node
 * :param num_entries: the number of elements (NOT BYTES!) that should
 * be loaded
 * :param data_type: the datatype of :var:`data`
 * :param shape: an array of size :var:`dimensions`, each entry describing the
 * size of the corresponding dimension. Make sure, :var:`data` is at least as
 * large as the product of all entries in :var:`shape`
 * :param dimensions: the number of dimensions
 *
 * Creates a Graph with a single store instruction, the data is
 * copied to intern memory, so after return of the function, :var:`data` does
 * not have to stay valid. :var:`shape` is copied as well. */
FGraphNode *fCreateGraph(const void *data, const int num_entries,
                         const FType data_type, const size_t *shape,
                         const int dimensions);

/** Creates a tensor that contains the single given values in all entries
 *
 * @param value the value this tensor should consist of
 * @param shape an array of size \p dimensions, each entry describing the size
 * of the corresponding dimension.
 * @param dimensions the number of dimensions
 */
FGraphNode *fconstant_i(const int value, const size_t *shape,
                        const int dimensions);
FGraphNode *fconstant_l(const long value, const size_t *shape,
                        const int dimensions);
FGraphNode *fconstant_f(const float value, const size_t *shape,
                        const int dimensions);
FGraphNode *fconstant_d(const double value, const size_t *shape,
                        const int dimensions);

/** Decrements :member:`FGraphNode.reference_counter` of :var:`graph` and
 * deallocates the node and its corresponding data, if the counter becomes 0. If
 * the node is deallocated, the same process is repeated with its predecessors.
 * So you can safely connect nodes multiple times and have only to free the leaf
 * nodes (i.e. the results), without caring about cross-reference, since thouse
 * are handles by the reference counting system.*/
void fFreeGraph(FGraphNode *graph);

/** Copies the graph node, the corresponding operation and additional data and
 * the predecessors (their :member:`FGraphNode.reference_counter` is
 * incremented) */
FGraphNode *fCopyGraph(const FGraphNode *graph);

/* Executes the graph node operations from all yet to be executed predecessors
 * to :var:`node` and returns a node with a :struct:`FResultData` operation in
 * which the resulting data is stored. If the graph is executed by the GPU
 * backend, a opencl kernel containing all selected operations is compiled and
 * executed. The kernels are cashed, so it improves the performance of a program
 * if the same graph-structures are reused (not necessary the same nodes, but
 * the same combination of nodes), since then the backend can reuse already
 * compiled kernels. If the CPU backend is chosen, it does not matter, since
 * every operation is executed independently.*/
FGraphNode *fExecuteGraph(FGraphNode *node);
FGraphNode *fExecuteGraph_cpu(FGraphNode *node);
FGraphNode *fExecuteGraph_gpu(FGraphNode *node);
FGraphNode *fExecuteGraph_cpu_eagerly(FGraphNode *node);
FGraphNode *fExecuteGraph_gpu_eagerly(FGraphNode *node);
//  operations
/** Elementwise addition of a and b :math:`a+b`. */
FGraphNode *fadd_g(FGraphNode *a, FGraphNode *b);
FGraphNode *fsub_g(FGraphNode *a, FGraphNode *b);
FGraphNode *fdiv_g(FGraphNode *a, FGraphNode *b);
FGraphNode *fmul_g(FGraphNode *a, FGraphNode *b);
FGraphNode *fpow_g(FGraphNode *a, FGraphNode *b);

/** Elementwise addition of a and b :math:`a+b`. */
FGraphNode *fadd_ci(FGraphNode *a, const int b);
FGraphNode *fadd_cl(FGraphNode *a, const long b);
FGraphNode *fadd_cf(FGraphNode *a, const float b);
FGraphNode *fadd_cd(FGraphNode *a, const double b);

/** Elementwise subtraction of a and b :math:`a-b`. */
FGraphNode *fsub_ci(FGraphNode *a, const int b);
FGraphNode *fsub_cl(FGraphNode *a, const long b);
FGraphNode *fsub_cf(FGraphNode *a, const float b);
FGraphNode *fsub_cd(FGraphNode *a, const double b);

/** Elementwise division of a and b :math:`\frac{a}{b}`. */
FGraphNode *fdiv_ci(FGraphNode *a, const int b);
FGraphNode *fdiv_cl(FGraphNode *a, const long b);
FGraphNode *fdiv_cf(FGraphNode *a, const float b);
FGraphNode *fdiv_cd(FGraphNode *a, const double b);

/** Elementwise multiplication of a and b :math:`a\cdot b`. */
FGraphNode *fmul_ci(FGraphNode *a, const int b);
FGraphNode *fmul_cl(FGraphNode *a, const long b);
FGraphNode *fmul_cf(FGraphNode *a, const float b);
FGraphNode *fmul_cd(FGraphNode *a, const double b);

/** Takes the elementwise power of a to b: :math:`a^b`.*/
FGraphNode *fpow_ci(FGraphNode *a, const int b);
FGraphNode *fpow_cl(FGraphNode *a, const long b);
FGraphNode *fpow_cf(FGraphNode *a, const float b);
FGraphNode *fpow_cd(FGraphNode *a, const double b);

FGraphNode *fgradient_add_g(const FGraphNode *a, const FGraphNode *b,
                            const FGraphNode *dx);
FGraphNode *fgradient_sub_g(const FGraphNode *a, const FGraphNode *b,
                            const FGraphNode *dx);
FGraphNode *fgradient_mul_g(const FGraphNode *a, const FGraphNode *b,
                            const FGraphNode *dx);

/** Carries out matrix multiplication on the last two dimensions of the tensors.
E.g. a matrix multiplication of two tensors with shapes (64, 32, 16) and (16,
24) will yield a tensor with shape (64, 32, 24). Since for one entry of the
tensor multiple other previous entries are needed, the operand tensors need to
be executed first. Therefor the method will implicitly (or eagerly) execute the
two parameter nodes if their data is not allready present, the given pointers
will be overwritten with the results. */
FGraphNode *fmatmul(FGraphNode **a, FGraphNode **b);

/** The first method flattens the complete tensor to a tensor with one
dimension, the second method flattens the tensor with :math:`n` dimensions along
:c:var:`dimension`, resulting in a tensor with :math:`n-1` dimensions.
Flattening a dimension will remove it from the shape of the tensor, therefor its
not possible to flatten the dimension 0.
A Tensor :math:`[[[3, 1, 4], [2, 1, 5]], [[0, 4, 2], [4, 7, 9]]]` flattened
along dimension 1 will result in :math:`[[3,1,4], [2,1,5], [0,4,2], [4,7,9]]`.
*/
FGraphNode *fflatten(FGraphNode *a);
FGraphNode *fflatten_dimension(FGraphNode *a, int dimension);

/** Converts the data of :var:`a` to the type given by :var:`newtype`*/
FGraphNode *fconvert(FGraphNode *a, FType newtype);
/** Reshapes the underlying data of the tensor to the new shape. The product of
  each dimension of the new shape must be the same as the product of the
  dimensions of the previous shape (i.e. it must describe the same number of
  entries of the tensor).*/
FGraphNode *freshape(FGraphNode *a, size_t *newshape, int dimensions);
// takes the minimum of two tensors element wise along the last dimension of
// each
FGraphNode *fmin_g(FGraphNode *a, FGraphNode *b);
FGraphNode *fmin_ci(FGraphNode *a, const int b);
FGraphNode *fmin_cl(FGraphNode *a, const long b);
FGraphNode *fmin_cf(FGraphNode *a, const float b);
FGraphNode *fmin_cd(FGraphNode *a, const double b);
// takes the maximum of two tensors element wise along the last dimension of
// each
FGraphNode *fmax_g(FGraphNode *a, FGraphNode *b);
FGraphNode *fmax_ci(FGraphNode *a, const int b);
FGraphNode *fmax_cl(FGraphNode *a, const long b);
FGraphNode *fmax_cf(FGraphNode *a, const float b);
FGraphNode *fmax_cd(FGraphNode *a, const double b);
/** Reduces one dimension of the tensor by additive folding e.g.
 * freduce_sum([[1,2,3], [4,5,6]], 0) = [5,7,9],
 * freduce_sum([[1,2,3], [4,5,6]], 1) = [6,15]
 * The results of the predecessor node must be available; to
 * ensure that the method may execute the parameter node. The corresponding
 * result node is then stored in the memory pointed to by a. */
FGraphNode *freduce_sum(FGraphNode **a, const int dimension);
/** Reduces one dimension of the tensor by multiplicative folding e.g.
 * freduce_mul([[1,2,3], [4,5,6]], 0) = [4,10,18],
 * freduce_mul([[1,2,3], [4,5,6]], 1) = [6, 120]
 * The results of the predecessor node must be available; to
 * ensure that the method may execute the parameter node. The corresponding
 * result node is then stored in the memory pointed to by a. */
FGraphNode *freduce_mul(FGraphNode **a, const int dimension);
/** Selects a slice of the tensor with a dimension wise start and end index.
 * :var:`start` and :var:`end` are arrays with as many entries
 * as the tensor has dimensions. They may contain negative values,
 * which are then subtracted from the end of the tensor (e.g. -1 means the last
 * element). :var:`start` is inclusive and describes the start index of the
 * selection per dimension and :var:`end` describes the end index per dimension
 * and is inclusive as well.
 */
FGraphNode *fslice(FGraphNode *a, const long *start, const long *end);
/** Selects a slice of the tensor with a dimension wise start index, end index
 * and step size. :var:`start`, :var:`end` and :var:`step` are arrays with as
 * many entries as the tensor has dimensions. :var:`start` and :var:`end` may
 * contain negative values, which are then subtracted from the end of the tensor
 * (e.g. -1 means the last element). :var:`start` is inclusive and describes the
 * start index of the selection per dimension and :var:`end` describes the end
 * index per dimension and is inclusive as well. :var:`step` contains the per
 * dimension step size (2 meaning every second element will be selected etc.)
 * and may be negative as well, which reverses the traversal order (the first
 * elements are selected as the last ones). For a negative step size,
 * :var:`start` > :var:`end` must hold (for a positive of course :var:`end` >
 * :var:`start`) for each dimension.
 */
FGraphNode *fslice_step(FGraphNode *a, const long *start, const long *end,
                        const long *step);

FGraphNode *fabs_g(FGraphNode *a);

#ifdef __cplusplus
}
// no c++ bindings, but function overloading for c++ header
inline FGraphNode *fconstant(const int value, const size_t *shape,
                             const int dimensions) {
  return fconstant_i(value, shape, dimensions);
}
inline FGraphNode *fconstant(const long value, const size_t *shape,
                             const int dimensions) {
  return fconstant_l(value, shape, dimensions);
}
inline FGraphNode *fconstant(const float value, const size_t *shape,
                             const int dimensions) {
  return fconstant_f(value, shape, dimensions);
}
inline FGraphNode *fconstant(const double value, const size_t *shape,
                             const int dimensions) {
  return fconstant_d(value, shape, dimensions);
}

inline FGraphNode *fadd(FGraphNode *a, FGraphNode *b) { return fadd_g(a, b); }
inline FGraphNode *fadd(FGraphNode *a, const int b) { return fadd_ci(a, b); }
inline FGraphNode *fadd(FGraphNode *a, const long b) { return fadd_cl(a, b); }
inline FGraphNode *fadd(FGraphNode *a, const float b) { return fadd_cf(a, b); }
inline FGraphNode *fadd(FGraphNode *a, const double b) { return fadd_cd(a, b); }

inline FGraphNode *fsub(FGraphNode *a, FGraphNode *b) { return fsub_g(a, b); }
inline FGraphNode *fsub(FGraphNode *a, const int b) { return fsub_ci(a, b); }
inline FGraphNode *fsub(FGraphNode *a, const long b) { return fsub_cl(a, b); }
inline FGraphNode *fsub(FGraphNode *a, const float b) { return fsub_cf(a, b); }
inline FGraphNode *fsub(FGraphNode *a, const double b) { return fsub_cd(a, b); }

inline FGraphNode *fmul(FGraphNode *a, FGraphNode *b) { return fmul_g(a, b); }
inline FGraphNode *fmul(FGraphNode *a, const int b) { return fmul_ci(a, b); }
inline FGraphNode *fmul(FGraphNode *a, const long b) { return fmul_cl(a, b); }
inline FGraphNode *fmul(FGraphNode *a, const float b) { return fmul_cf(a, b); }
inline FGraphNode *fmul(FGraphNode *a, const double b) { return fmul_cd(a, b); }

inline FGraphNode *fdiv(FGraphNode *a, FGraphNode *b) { return fdiv_g(a, b); }
inline FGraphNode *fdiv(FGraphNode *a, const int b) { return fdiv_ci(a, b); }
inline FGraphNode *fdiv(FGraphNode *a, const long b) { return fdiv_cl(a, b); }
inline FGraphNode *fdiv(FGraphNode *a, const float b) { return fdiv_cf(a, b); }
inline FGraphNode *fdiv(FGraphNode *a, const double b) { return fdiv_cd(a, b); }

inline FGraphNode *fpow(FGraphNode *a, FGraphNode *b) { return fpow_g(a, b); }
inline FGraphNode *fpow(FGraphNode *a, const int b) { return fpow_ci(a, b); }
inline FGraphNode *fpow(FGraphNode *a, const long b) { return fpow_cl(a, b); }
inline FGraphNode *fpow(FGraphNode *a, const float b) { return fpow_cf(a, b); }
inline FGraphNode *fpow(FGraphNode *a, const double b) { return fpow_cd(a, b); }

inline FGraphNode *fmin(FGraphNode *a, FGraphNode *b) { return fmin_g(a, b); }
inline FGraphNode *fmin(FGraphNode *a, const int b) { return fmin_ci(a, b); }
inline FGraphNode *fmin(FGraphNode *a, const long b) { return fmin_cl(a, b); }
inline FGraphNode *fmin(FGraphNode *a, const float b) { return fmin_cf(a, b); }
inline FGraphNode *fmin(FGraphNode *a, const double b) { return fmin_cd(a, b); }

inline FGraphNode *fmax(FGraphNode *a, FGraphNode *b) { return fmax_g(a, b); }
inline FGraphNode *fmax(FGraphNode *a, const int b) { return fmax_ci(a, b); }
inline FGraphNode *fmax(FGraphNode *a, const long b) { return fmax_cl(a, b); }
inline FGraphNode *fmax(FGraphNode *a, const float b) { return fmax_cf(a, b); }
inline FGraphNode *fmax(FGraphNode *a, const double b) { return fmax_cd(a, b); }

inline FGraphNode *fflatten(FGraphNode *a, int dimension) {
  return fflatten_dimension(a, dimension);
}
#include <string>
inline void flog(FLogType type, std::string msg) { flog(type, msg.c_str()); }
#endif
#endif
