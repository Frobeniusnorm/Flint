#include "../flint.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include <cstring>
#include <stdlib.h>
#include <unordered_set>
#include <vector>
using namespace FlintBackend;

GraphNode *FlintBackend::createGraph(void *data, int num_entries,
                                     Type data_type, int *shape,
                                     int dimensions) {
  GraphNode *gn = new GraphNode();
  Store *op = new Store();
  op->dimensions = dimensions;
  op->shape = (int *)malloc(sizeof(int) * dimensions);
  std::memcpy((void *)op->shape, (void *)shape, dimensions * sizeof(int));
  op->data = data;
  op->num_entries = num_entries;
  op->data_type = data_type;
  gn->operation = op;
  gn->num_predecessor = 0;
  gn->predecessors = NULL;
  return gn;
}
static void collectAllNodes(GraphNode *graph,
                            std::unordered_set<GraphNode *> &set) {
  set.insert(graph);
  for (int i = 0; i < graph->num_predecessor; i++) {
    if (graph->predecessors[i] && set.find(graph->predecessors[i]) == set.end())
      collectAllNodes(graph->predecessors[i], set);
  }
}
// frees all allocated data from the graph and the nodes that are reachable
void FlintBackend::freeGraph(GraphNode *graph) {
  std::unordered_set<GraphNode *> all;
  collectAllNodes(graph, all);
  for (GraphNode *gn : all) {
    // delete gn->operation;
    if (gn->predecessors != NULL && gn->num_predecessor != 0)
      free(gn->predecessors);
    if (gn->operation != NULL) {
      if (gn->operation->shape)
        free(gn->operation->shape);
      if (gn->operation->op_type == RESULTDATA) {
        ResultData *rd = (ResultData *)gn->operation;
        free(rd->data);
      }
      delete gn->operation;
    }
    delete gn;
  }
}
// function to add nodes to the graph i.e. operations
static GraphNode *addNode(Operation *op, std::vector<GraphNode *> pre) {
  if (!op) {
    log(WARNING, "You are adding a node with a NULL operation, this is not "
                 "correct behaviour!");
  }
  GraphNode *foo = new GraphNode();
  foo->operation = op;
  foo->num_predecessor = pre.size();
  foo->predecessors =
      pre.size() == 0 ? NULL : safe_mal<GraphNode *>(pre.size());
  for (int i = 0; i < pre.size(); i++) {
    foo->predecessors[i] = pre[i];
  }
  return foo;
}
static inline void initShape_keep(Operation *op, Operation *a, Operation *b) {
  int *src = nullptr;
  if (!b || a->dimensions >= b->dimensions) {
    op->dimensions = a->dimensions;
    src = a->shape;
  } else {
    op->dimensions = b->dimensions;
    src = b->shape;
  }
  op->shape = (int *)malloc(sizeof(int) * op->dimensions);
  memcpy((void *)op->shape, src, sizeof(int) * op->dimensions);
  // determine type
  Type highest = INT32;
  if (a->data_type == FLOAT64 || (b && b->data_type == FLOAT64))
    highest = FLOAT64;
  else if (a->data_type == FLOAT32 || (b && b->data_type == FLOAT32))
    highest = FLOAT32;
  else if (a->data_type == INT64 || (b && b->data_type == INT64))
    highest = INT64;
  op->data_type = highest;
}
GraphNode *FlintBackend::add(GraphNode *a, GraphNode *b) {
  Add *op = new Add();
  op->op_type = ADD;
  initShape_keep(op, a->operation, b->operation);
  return addNode(op, {a, b});
}
GraphNode *FlintBackend::sub(GraphNode *a, GraphNode *b) {
  Sub *op = new Sub();
  op->op_type = SUB;
  initShape_keep(op, a->operation, b->operation);
  return addNode(op, {a, b});
}
GraphNode *FlintBackend::div(GraphNode *a, GraphNode *b) {
  Div *op = new Div();
  op->op_type = DIV;
  initShape_keep(op, a->operation, b->operation);
  return addNode(op, {a, b});
}
GraphNode *FlintBackend::mul(GraphNode *a, GraphNode *b) {
  Mul *op = new Mul();
  op->op_type = MUL;
  initShape_keep(op, a->operation, b->operation);
  return addNode(op, {a, b});
}
template <typename T>
static GraphNode *addNodeWithConst(Operation *op, GraphNode *a, T b) {
  Const<T> *cons = new Const<T>();
  cons->value = b;
  cons->op_type = CONST;
  if (typeid(T) == typeid(int))
    cons->data_type = INT32;
  else if (typeid(T) == typeid(long))
    cons->data_type = INT64;
  else if (typeid(T) == typeid(float))
    cons->data_type = FLOAT32;
  else if (typeid(T) == typeid(double))
    cons->data_type = FLOAT64;
  return addNode(op, {a, addNode(cons, {})});
}
// adds the constant value to each entry in a
template <typename T> GraphNode *FlintBackend::add(GraphNode *a, T b) {
  Add *op = new Add();
  op->op_type = ADD;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
template GraphNode *FlintBackend::add<double>(GraphNode *, double);
template GraphNode *FlintBackend::add<float>(GraphNode *, float);
template GraphNode *FlintBackend::add<int>(GraphNode *, int);
template GraphNode *FlintBackend::add<long>(GraphNode *, long);
// subtracts the constant value from each entry in a
template <typename T> GraphNode *FlintBackend::sub(GraphNode *a, T b) {
  Sub *op = new Sub();
  op->op_type = SUB;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
template GraphNode *FlintBackend::sub<double>(GraphNode *, double);
template GraphNode *FlintBackend::sub<float>(GraphNode *, float);
template GraphNode *FlintBackend::sub<int>(GraphNode *, int);
template GraphNode *FlintBackend::sub<long>(GraphNode *, long);
// divides each entry in a by the constant value
template <typename T> GraphNode *FlintBackend::div(GraphNode *a, T b) {
  Div *op = new Div();
  op->op_type = DIV;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
template GraphNode *FlintBackend::div<double>(GraphNode *, double);
template GraphNode *FlintBackend::div<float>(GraphNode *, float);
template GraphNode *FlintBackend::div<int>(GraphNode *, int);
template GraphNode *FlintBackend::div<long>(GraphNode *, long);
// multiplicates the constant value with each entry in a
template <typename T> GraphNode *FlintBackend::mul(GraphNode *a, T b) {
  Mul *op = new Mul();
  op->op_type = MUL;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
template GraphNode *FlintBackend::mul<double>(GraphNode *, double);
template GraphNode *FlintBackend::mul<float>(GraphNode *, float);
template GraphNode *FlintBackend::mul<int>(GraphNode *, int);
template GraphNode *FlintBackend::mul<long>(GraphNode *, long);

FlintBackend::ResultData::~ResultData() {
  if (this->data)
    free(this->data);
}
