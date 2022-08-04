#ifndef FLINT_H
#define FLINT_H
#include "src/logger.hpp"
#define CL_TARGET_OPENCL_VERSION 200
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
/*
  This is the basic header file and implementation of Flint, to be compatible
  with as many languages as possible. Though it is written in C++ for object
  orientation it is as close to C as possible to guarantee compatibility,
  nevertheless it is NOT compatible to C. Maybe in the future i will rewrite
  this header to add C compatibility.

  Flints Execution structure represents an AST, so that each Graph may be
  compiled to a specific OpenCL program
*/

void flintInit();
void flintCleanup();
// 0 no logging, 1 only errors, 2 errors and warnings, 3 error, warnings and
// info, 4 verbose
void setLoggingLevel(int);

enum FType { FLOAT32, FLOAT64, INT32, INT64 };
enum FOperationType { STORE, RESULTDATA, CONST, ADD, SUB, MUL, DIV, Length };
struct FOperation {
  // shape of the data after execution
  int dimensions;
  int *shape;
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
  int kernel_tree_id; // internal node id of current operation in kernel caching
                      // tree
  FGraphNode **predecessors;
  FOperation *operation; // the operation represented by this graph node
};
// Operations
// Store data in the tensor, always an Entry (source) of the graph
struct FStore {
  // link to gpu data
  cl_mem mem_id = nullptr;
  void *data;
  int num_entries;
};
struct FResultData {
  // link to gpu data
  cl_mem mem_id = nullptr;
  void *data;
  int num_entries;
};
// A single-value constant (does not have to be loaded as a tensor)
struct FConst {
  void *value; // has to be one of Type
};

// functions
// creates a Graph with a single store instruction, data is copied _during
// execution_ to intern memory, till then the pointer has to point to
// valid data. Shape is instantly copied and has to be alive only during this
// function call. Data contains the flattened data array from type data_type and
// shape contains the size on each dimension as an array with length dimensions.
FGraphNode *createGraph(void *data, int num_entries, FType data_type,
                        int *shape, int dimensions);
// frees all allocated data from the graph and the nodes that are reachable
void freeGraph(FGraphNode *graph);
/* executes the Graph which starts with the root of the given node and stores
 * the result for that node in result. If Flint was not initialized yet, this
 * function will do that automatically.
 * The result data is automatically freed with the Graph */
FGraphNode *executeGraph(FGraphNode *node);
// operations
FGraphNode *add(FGraphNode *a, FGraphNode *b);
FGraphNode *sub(FGraphNode *a, FGraphNode *b);
FGraphNode *div(FGraphNode *a, FGraphNode *b);
FGraphNode *mul(FGraphNode *a, FGraphNode *b);
// adds the constant value to each entry in a
FGraphNode *add(FGraphNode *a, int b);
// subtracts the constant value from each entry in a
FGraphNode *sub(FGraphNode *a, long b);
// divides each entry in a by the constant value
FGraphNode *div(FGraphNode *a, float b);
// multiplicates the constant value with each entry in a
FGraphNode *mul(FGraphNode *a, double b);
#endif
