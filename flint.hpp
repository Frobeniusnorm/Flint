#ifndef FLINT_H
#define FLINT_H
#include "src/logger.hpp"
#include <CL/cl.h>
/*
  This is the basic header file and implementation of Flint, to be compatible
  with as many languages as possible. Though it is written in C++ for object
  orientation it is as close to C as possible to guarantee compatibility,
  nevertheless it is NOT compatible to C. Maybe in the future i will rewrite
  this header to add C compatibility.

  Flints Execution structure represents an AST, so that each Graph may be
  compiled to a specific OpenCL program
*/
namespace FlintBackend {

void init();
void cleanup();
// 0 no logging, 1 only errors, 2 errors and warnings, 3 error, warnings and
// info, 4 verbose
void setLoggingLevel(int);

enum Type { FLOAT32, FLOAT64, INT32, INT64 };
enum OperationType { STORE, RESULTDATA, CONST, ADD, SUB, MUL, DIV };
struct Operation {
  // link to gpu data
  cl_mem mem_id = nullptr;
  // shape of the data after execution
  int dimensions;
  int *shape;
  // type of operation, to enable switch cases and avoid v-table lookups
  OperationType op_type;
  // datatype of result
  Type data_type;
};
// each Graph Node represents one Operation with multiple ingoing edges and one
// out going edge
struct GraphNode {
  int num_predecessor;
  GraphNode **predecessors, *successor;
  Operation *operation; // the operation represented by this graph node
};
// Operations
// Store data in the tensor, always an Entry (source) of the graph
struct Store : public Operation {
  void *data;
  int num_entries;
};
struct ResultData : public Operation {
  void *data;
  int num_entries;
  ~ResultData();
};
// Addition of two tensors
struct Add : public Operation {};
// Subtraction of two tensors
struct Sub : public Operation {};
// Multiplication of two tensors
struct Mul : public Operation {};
// Division of two tensors
struct Div : public Operation {};
// A single-value constant (does not have to be loaded as a tensor)
template <typename T> struct Const : public Operation {
  T value; // has to be one of Type
};

// functions
// creates a Graph with a single store instruction, data is copied _during
// execution_ to intern memory, till then the pointer has to point to
// valid data. Shape is instantly copied and has to be alive only during this
// function call. Data contains the flattened data array from type data_type and
// shape contains the size on each dimension as an array with length dimensions.
GraphNode *createGraph(void *data, int num_entries, Type data_type, int *shape,
                       int dimensions);
// frees all allocated data from the graph and the nodes that are reachable
void freeGraph(GraphNode *graph);
/* executes the Graph which starts with the root of the given node and stores
 * the result for that node in result. If Flint was not initialized yet, this
 * function will do that automatically.
 * The result data is automatically freed with the Graph */
ResultData *executeGraph(GraphNode *node);
// operations
GraphNode *add(GraphNode *a, GraphNode *b);
GraphNode *sub(GraphNode *a, GraphNode *b);
GraphNode *div(GraphNode *a, GraphNode *b);
GraphNode *mul(GraphNode *a, GraphNode *b);
// adds the constant value to each entry in a
template <typename T> GraphNode *add(GraphNode *a, T b);
// subtracts the constant value from each entry in a
template <typename T> GraphNode *sub(GraphNode *a, T b);
// divides each entry in a by the constant value
template <typename T> GraphNode *div(GraphNode *a, T b);
// multiplicates the constant value with each entry in a
template <typename T> GraphNode *mul(GraphNode *a, T b);
}; // namespace FlintBackend
#endif
