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
#include "src/logger.hpp"
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
void setLoggingLevel(int);

enum FType { INT32, INT64, FLOAT32, FLOAT64 };
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
  Length
};
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
// each Graph Node represents one Operation with multiple ingoing edges and one
// out going edge
struct FGraphNode {
  int num_predecessor;
  FGraphNode **predecessors;
  FOperation *operation;    // the operation represented by this graph node
  size_t reference_counter; // for garbage collection in free graph
};
// Operations
// Store data in the tensor, always an Entry (source) of the graph
struct FStore {
  // link to gpu data
  cl_mem mem_id = nullptr;
  void *data;
  size_t num_entries;
};
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

// functions

// creates a Graph with a single store instruction, data is copied to intern
// memory, so after return of the function, data may be deleted. Shape is also
// instantly copied. Data contains the flattened data array from type data_type
// and shape contains the size on each dimension as an array with length
// dimensions.
FGraphNode *fCreateGraph(void *data, int num_entries, FType data_type,
                         size_t *shape, int dimensions);
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
#ifdef __cplusplus
}
// no c++ bindings, but function overloading for c++ header
inline FGraphNode *add(FGraphNode *a, FGraphNode *b) { return fadd_g(a, b); }
inline FGraphNode *add(FGraphNode *a, const int b) { return fadd_ci(a, b); }
inline FGraphNode *add(FGraphNode *a, const long b) { return fadd_cl(a, b); }
inline FGraphNode *add(FGraphNode *a, const float b) { return fadd_cf(a, b); }
inline FGraphNode *add(FGraphNode *a, const double b) { return fadd_cd(a, b); }

inline FGraphNode *sub(FGraphNode *a, FGraphNode *b) { return fsub_g(a, b); }
inline FGraphNode *sub(FGraphNode *a, const int b) { return fsub_ci(a, b); }
inline FGraphNode *sub(FGraphNode *a, const long b) { return fsub_cl(a, b); }
inline FGraphNode *sub(FGraphNode *a, const float b) { return fsub_cf(a, b); }
inline FGraphNode *sub(FGraphNode *a, const double b) { return fsub_cd(a, b); }

inline FGraphNode *mul(FGraphNode *a, FGraphNode *b) { return fmul_g(a, b); }
inline FGraphNode *mul(FGraphNode *a, const int b) { return fmul_ci(a, b); }
inline FGraphNode *mul(FGraphNode *a, const long b) { return fmul_cl(a, b); }
inline FGraphNode *mul(FGraphNode *a, const float b) { return fmul_cf(a, b); }
inline FGraphNode *mul(FGraphNode *a, const double b) { return fmul_cd(a, b); }

inline FGraphNode *div(FGraphNode *a, FGraphNode *b) { return fdiv_g(a, b); }
inline FGraphNode *div(FGraphNode *a, const int b) { return fdiv_ci(a, b); }
inline FGraphNode *div(FGraphNode *a, const long b) { return fdiv_cl(a, b); }
inline FGraphNode *div(FGraphNode *a, const float b) { return fdiv_cf(a, b); }
inline FGraphNode *div(FGraphNode *a, const double b) { return fdiv_cd(a, b); }

inline FGraphNode *pow(FGraphNode *a, FGraphNode *b) { return fpow_g(a, b); }
inline FGraphNode *pow(FGraphNode *a, const int b) { return fpow_ci(a, b); }
inline FGraphNode *pow(FGraphNode *a, const long b) { return fpow_cl(a, b); }
inline FGraphNode *pow(FGraphNode *a, const float b) { return fpow_cf(a, b); }
inline FGraphNode *pow(FGraphNode *a, const double b) { return fpow_cd(a, b); }

inline FGraphNode *flatten(FGraphNode *a, int dimension) {
  return fflatten_dimension(a, dimension);
}
inline FGraphNode *flatten(FGraphNode *a) { return fflatten(a); }
#endif
#endif
