#include "../flint.h"
#include "logger.hpp"
#include "utils.hpp"
#include <cstring>
#include <stdlib.h>
#include <unordered_set>
#include <vector>

FGraphNode *createGraph(void *data, int num_entries, FType data_type,
                        int *shape, int dimensions) {
  FGraphNode *gn = new FGraphNode();
  FOperation *op = new FOperation();
  FStore *store = new FStore();
  op->dimensions = dimensions;
  op->shape = (int *)malloc(sizeof(int) * dimensions);
  std::memcpy((void *)op->shape, (void *)shape, dimensions * sizeof(int));
  op->additional_data = (void *)store;
  op->op_type = STORE;
  store->data = data;
  store->num_entries = num_entries;
  op->data_type = data_type;
  gn->operation = op;
  gn->num_predecessor = 0;
  gn->predecessors = NULL;
  return gn;
}
static void collectAllNodes(FGraphNode *graph,
                            std::unordered_set<FGraphNode *> &set) {
  set.insert(graph);
  for (int i = 0; i < graph->num_predecessor; i++) {
    if (graph->predecessors[i] && set.find(graph->predecessors[i]) == set.end())
      collectAllNodes(graph->predecessors[i], set);
  }
}
// frees all allocated data from the graph and the nodes that are reachable
void freeGraph(FGraphNode *graph) {
  std::unordered_set<FGraphNode *> all;
  collectAllNodes(graph, all);
  for (FGraphNode *gn : all) {
    if (gn->predecessors != NULL && gn->num_predecessor != 0)
      free(gn->predecessors);
    if (gn->operation != NULL) {
      if (gn->operation->shape)
        free(gn->operation->shape);
      if (gn->operation->additional_data)
        switch (gn->operation->op_type) {
        case RESULTDATA: {
          FResultData *rd = (FResultData *)gn->operation->additional_data;
          if (rd->data)
            free(rd->data);
          delete rd;
        } break;
        case STORE:
          delete (FStore *)gn->operation->additional_data;
          break;
        case CONST: {
          FConst *c = (FConst *)gn->operation->additional_data;
          free(c->value);
          delete c;
        } break;
        default:
          break;
        }
      delete gn->operation;
    }
    delete gn;
  }
}
// function to add nodes to the graph i.e. operations
static FGraphNode *addNode(FOperation *op, std::vector<FGraphNode *> pre) {
  if (!op) {
    log(WARNING, "You are adding a node with a NULL operation, this is not "
                 "correct behaviour!");
  }
  FGraphNode *foo = new FGraphNode();
  foo->operation = op;
  foo->num_predecessor = pre.size();
  foo->predecessors =
      pre.size() == 0 ? NULL : safe_mal<FGraphNode *>(pre.size());
  for (int i = 0; i < pre.size(); i++) {
    foo->predecessors[i] = pre[i];
  }
  return foo;
}
static inline void initShape_keep(FOperation *op, FOperation *a,
                                  FOperation *b) {
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
  FType highest = INT32;
  if (a->data_type == FLOAT64 || (b && b->data_type == FLOAT64))
    highest = FLOAT64;
  else if (a->data_type == FLOAT32 || (b && b->data_type == FLOAT32))
    highest = FLOAT32;
  else if (a->data_type == INT64 || (b && b->data_type == INT64))
    highest = INT64;
  op->data_type = highest;
}
FGraphNode *add(FGraphNode *a, FGraphNode *b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = ADD;
  initShape_keep(op, a->operation, b->operation);
  return addNode(op, {a, b});
}
FGraphNode *sub(FGraphNode *a, FGraphNode *b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = SUB;
  initShape_keep(op, a->operation, b->operation);
  return addNode(op, {a, b});
}
FGraphNode *div(FGraphNode *a, FGraphNode *b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = DIV;
  initShape_keep(op, a->operation, b->operation);
  return addNode(op, {a, b});
}
FGraphNode *mul(FGraphNode *a, FGraphNode *b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = MUL;
  initShape_keep(op, a->operation, b->operation);
  return addNode(op, {a, b});
}
FGraphNode *pow(FGraphNode *a, FGraphNode *b) {
  if (!(a->operation->dimensions >= b->operation->dimensions))
    log(ERROR, "pow(a, b) must fulfill a->dimensions >= b->dimensions");
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = POW;
  initShape_keep(op, a->operation, b->operation);
  return addNode(op, {a, b});
}
template <typename T>
static FGraphNode *addNodeWithConst(FOperation *op, FGraphNode *a, T b) {
  FConst *cons = new FConst();
  T *cons_val = (T *)malloc(sizeof(T));
  *cons_val = b;
  cons->value = (void *)cons_val;
  FOperation *cop = new FOperation();
  cop->op_type = CONST;
  cop->additional_data = (void *)cons;
  if (typeid(T) == typeid(int))
    cop->data_type = INT32;
  else if (typeid(T) == typeid(long))
    cop->data_type = INT64;
  else if (typeid(T) == typeid(float))
    cop->data_type = FLOAT32;
  else if (typeid(T) == typeid(double))
    cop->data_type = FLOAT64;
  return addNode(op, {a, addNode(cop, {})});
}
// adds the constant value to each entry in a
template <typename T> FGraphNode *add(FGraphNode *a, T b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = ADD;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
FGraphNode *add(FGraphNode *a, double b) { return add<double>(a, b); }
FGraphNode *add(FGraphNode *a, float b) { return add<float>(a, b); }
FGraphNode *add(FGraphNode *a, int b) { return add<int>(a, b); }
FGraphNode *add(FGraphNode *a, long b) { return add<long>(a, b); }
// subtracts the constant value from each entry in a
template <typename T> FGraphNode *sub(FGraphNode *a, T b) {
  FOperation *op = new FOperation();
  op->op_type = SUB;
  op->additional_data = nullptr;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
FGraphNode *sub(FGraphNode *a, double b) { return sub<double>(a, b); }
FGraphNode *sub(FGraphNode *a, float b) { return sub<float>(a, b); }
FGraphNode *sub(FGraphNode *a, int b) { return sub<int>(a, b); }
FGraphNode *sub(FGraphNode *a, long b) { return sub<long>(a, b); }
// divides each entry in a by the constant value
template <typename T> FGraphNode *div(FGraphNode *a, T b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = DIV;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
FGraphNode *div(FGraphNode *a, double b) { return div<double>(a, b); }
FGraphNode *div(FGraphNode *a, float b) { return div<float>(a, b); }
FGraphNode *div(FGraphNode *a, int b) { return div<int>(a, b); }
FGraphNode *div(FGraphNode *a, long b) { return div<long>(a, b); }
// multiplicates the constant value with each entry in a
template <typename T> FGraphNode *mul(FGraphNode *a, T b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = MUL;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
FGraphNode *mul(FGraphNode *a, double b) { return mul<double>(a, b); }
FGraphNode *mul(FGraphNode *a, float b) { return mul<float>(a, b); }
FGraphNode *mul(FGraphNode *a, int b) { return mul<int>(a, b); }
FGraphNode *mul(FGraphNode *a, long b) { return mul<long>(a, b); }
// takes the power of each element in a to b
template <typename T> FGraphNode *pow(FGraphNode *a, T b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = MUL;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
FGraphNode *pow(FGraphNode *a, double b) { return pow<double>(a, b); }
FGraphNode *pow(FGraphNode *a, float b) { return pow<float>(a, b); }
FGraphNode *pow(FGraphNode *a, int b) { return pow<int>(a, b); }
FGraphNode *pow(FGraphNode *a, long b) { return pow<long>(a, b); }
