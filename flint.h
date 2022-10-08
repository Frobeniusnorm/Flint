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
  with as many languages as possible. It should be C compatible.

  Flints Execution structure represents an AST, so that each Graph may be
  compiled to a specific OpenCL program
*/

// initializes flint, if cpu is set the cpu backend is initialized and used, if
// gpu is set the gpu backend is initialized and used
void flintInit(int cpu, int gpu);
void flintInit_cpu();
void flintInit_gpu();
void flintCleanup();
void flintCleanup_cpu();
void flintCleanup_gpu();
// 0 no logging, 1 only errors, 2 errors and warnings, 3 error, warnings and
// info, 4 verbose
void fSetLoggingLevel(int);
enum FLogType { F_DEBUG, F_VERBOSE, F_INFO, F_ERROR, F_WARNING };
void flog(FLogType type, const char *msg);
enum FType { F_INT32, F_INT64, F_FLOAT32, F_FLOAT64 };
enum FOperationType {
  STORE,
  RESULTDATA,
  CONST,
  ADD,
  SUB,
  MUL,
  DIV,
  POW,
  NEG,
  FLATTEN,
  MATMUL,
  CONVERSION,
  RESHAPE,
  MIN,
  MAX,
  REDUCE_SUM,
  REDUCE_MUL,
  SLICE,
  Length
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
  size_t *start;
  size_t *end;
  size_t *step;
};
// functions

// creates a Graph with a single store instruction, data is copied to intern
// memory, so after return of the function, data may be deleted. Shape is also
// instantly copied. Data contains the flattened data array from type data_type
// and shape contains the size on each dimension as an array with length
// dimensions.
FGraphNode *fCreateGraph(const void *data, const int num_entries,
                         const FType data_type, const size_t *shape,
                         const int dimensions);
// frees all allocated data from the graph and the nodes that are reachable
void fFreeGraph(FGraphNode *graph);

FGraphNode *fCopyGraph(const FGraphNode *graph);
/* executes the Graph which starts with the root of the given node and stores
 * the result for that node in result. If Flint was not initialized yet, this
 * function will do that automatically.
 * This function will automatically determine if the graph is executed on the
 * cpu or on the gpu. The result data is automatically freed with the Graph */
FGraphNode *fExecuteGraph(FGraphNode *node);
FGraphNode *fExecuteGraph_cpu(FGraphNode *node);
FGraphNode *fExecuteGraph_gpu(FGraphNode *node);
// operations
FGraphNode *fadd_g(FGraphNode *a, FGraphNode *b);
FGraphNode *fsub_g(FGraphNode *a, FGraphNode *b);
FGraphNode *fdiv_g(FGraphNode *a, FGraphNode *b);
FGraphNode *fmul_g(FGraphNode *a, FGraphNode *b);
FGraphNode *fpow_g(FGraphNode *a, FGraphNode *b);
// adds the constant value to each entry in a
FGraphNode *fadd_ci(FGraphNode *a, const int b);
FGraphNode *fadd_cl(FGraphNode *a, const long b);
FGraphNode *fadd_cf(FGraphNode *a, const float b);
FGraphNode *fadd_cd(FGraphNode *a, const double b);
// subtracts the constant value from each entry in a
FGraphNode *fsub_ci(FGraphNode *a, const int b);
FGraphNode *fsub_cl(FGraphNode *a, const long b);
FGraphNode *fsub_cf(FGraphNode *a, const float b);
FGraphNode *fsub_cd(FGraphNode *a, const double b);
// divides each entry in a by the constant value
FGraphNode *fdiv_ci(FGraphNode *a, const int b);
FGraphNode *fdiv_cl(FGraphNode *a, const long b);
FGraphNode *fdiv_cf(FGraphNode *a, const float b);
FGraphNode *fdiv_cd(FGraphNode *a, const double b);
// multiplicates the constant value with each entry in a
FGraphNode *fmul_ci(FGraphNode *a, const int b);
FGraphNode *fmul_cl(FGraphNode *a, const long b);
FGraphNode *fmul_cf(FGraphNode *a, const float b);
FGraphNode *fmul_cd(FGraphNode *a, const double b);
// computes the power of each entry to the constant
FGraphNode *fpow_ci(FGraphNode *a, const int b);
FGraphNode *fpow_cl(FGraphNode *a, const long b);
FGraphNode *fpow_cf(FGraphNode *a, const float b);
FGraphNode *fpow_cd(FGraphNode *a, const double b);
// matrix multiplication, multiplication is carried out on the two inner most
// dimensions. The results of the two predecessors nodes must be available; to
// ensure that the method may execute the parameter nodes. The corresponding
// result nodes are then stored in the memory pointed to by a and b.
FGraphNode *fmatmul(FGraphNode **a, FGraphNode **b);
// flattens the GraphNode data to a 1 dimensional tensor, no additional data is
// allocated besides the new node
FGraphNode *fflatten(FGraphNode *a);
// flattens a specific dimension of the tensor, no additional data is allocated
// besides the new node
FGraphNode *fflatten_dimension(FGraphNode *a, int dimension);

// Converts the type
FGraphNode *fconvert(FGraphNode *a, FType newtype);
// reshapes the Tensor, requires execution of predecessor
FGraphNode *freshape(FGraphNode *a, size_t *newshape, int dimensions);
// takes the minimum of two tensors element wise
FGraphNode *fmin_g(FGraphNode *a, FGraphNode *b);
FGraphNode *fmin_ci(FGraphNode *a, const int b);
FGraphNode *fmin_cl(FGraphNode *a, const long b);
FGraphNode *fmin_cf(FGraphNode *a, const float b);
FGraphNode *fmin_cd(FGraphNode *a, const double b);
// takes the maximum of two tensors element wise
FGraphNode *fmax_g(FGraphNode *a, FGraphNode *b);
FGraphNode *fmax_ci(FGraphNode *a, const int b);
FGraphNode *fmax_cl(FGraphNode *a, const long b);
FGraphNode *fmax_cf(FGraphNode *a, const float b);
FGraphNode *fmax_cd(FGraphNode *a, const double b);
// reduces one dimension of the tensor by additive folding e.g.
// freduce_sum([[1,2,3], [4,5,6]], 0) = [5,7,9],
// freduce_sum([[1,2,3], [4,5,6]], 1) = [6,15]
// The results of the predecessor node must be available; to
// ensure that the method may execute the parameter node. The corresponding
// result node is then stored in the memory pointed to by a.
FGraphNode *freduce_sum(FGraphNode **a, const int dimension);
// reduces one dimension of the tensor by additive multiplication e.g.
// freduce_sum([[1,2,3], [4,5,6]], 0) = [4,10,18],
// freduce_sum([[1,2,3], [4,5,6]], 1) = [6, 120]
// The results of the predecessor node must be available; to
// ensure that the method may execute the parameter node. The corresponding
// result node is then stored in the memory pointed to by a.
FGraphNode *freduce_mul(FGraphNode **a, const int dimension);
/* Selects a slice of the tensor with a start and end range.
 * Start is inclusive end is exclusive.
 * @param start and @param end have to contain a index and size per dimension
 * of the target tensor. */
FGraphNode *fslice(FGraphNode *a, const size_t *start, const size_t *end);
/* Selects a slice of the tensor with a start and end range by a per dimension
 * step size. Start is inclusive end is exclusive.
 * @param start, @param end and @param step have to contain a index, size and
 * step size per dimension of the target tensor. */
FGraphNode *fslice_step(FGraphNode *a, const size_t *start, const size_t *end,
                        const size_t *step);

#ifdef __cplusplus
}
// no c++ bindings, but function overloading for c++ header
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
