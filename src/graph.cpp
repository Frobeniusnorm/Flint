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

#include "../flint.h"
#include "logger.hpp"
#include "utils.hpp"
#include <cstring>
#include <list>
#include <stdlib.h>
#include <unordered_set>
#include <vector>

FGraphNode *createGraph(void *data, int num_entries, FType data_type,
                        int *shape, int dimensions) {
  FGraphNode *gn = new FGraphNode();
  gn->reference_counter = 1;
  FOperation *op = new FOperation();
  FStore *store = new FStore();
  op->dimensions = dimensions;
  op->shape = (int *)malloc(sizeof(int) * dimensions);
  std::memcpy((void *)op->shape, (void *)shape, dimensions * sizeof(int));
  op->additional_data = (void *)store;
  op->op_type = STORE;
  size_t byte_size = num_entries;
  switch (data_type) {
  case INT32:
    store->data = safe_mal<int>(num_entries);
    byte_size *= sizeof(int);
    break;
  case INT64:
    store->data = safe_mal<long>(num_entries);
    byte_size *= sizeof(long);
    break;
  case FLOAT32:
    store->data = safe_mal<float>(num_entries);
    byte_size *= sizeof(float);
    break;
  case FLOAT64:
    store->data = safe_mal<long>(num_entries);
    byte_size *= sizeof(long);
    break;
  }
  memcpy(store->data, data, byte_size);
  store->num_entries = num_entries;
  op->data_type = data_type;
  gn->operation = op;
  gn->num_predecessor = 0;
  gn->predecessors = NULL;
  return gn;
}
// frees all allocated data from the graph and the nodes that are reachable
void freeGraph(FGraphNode *graph) {
  std::unordered_set<FGraphNode *> all;
  std::list<FGraphNode *> wq;
  graph->reference_counter--;
  all.insert(graph);
  wq.push_back(graph);
  while (!wq.empty()) {
    FGraphNode *gn = wq.front();
    wq.pop_front();
    all.erase(gn);
    if (gn->reference_counter != 0)
      continue;
    for (int i = 0; i < gn->num_predecessor; i++) {
      if (gn->predecessors[i] &&
          --(gn->predecessors[i]->reference_counter) == 0 &&
          all.find(gn->predecessors[i]) == all.end()) {
        wq.push_back(gn->predecessors[i]);
        all.insert(gn->predecessors[i]);
      }
    }
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
          free(((FStore *)gn->operation->additional_data)->data);
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
  foo->reference_counter = 1;
  foo->operation = op;
  foo->num_predecessor = pre.size();
  foo->predecessors =
      pre.size() == 0 ? NULL : safe_mal<FGraphNode *>(pre.size());
  for (int i = 0; i < pre.size(); i++) {
    foo->predecessors[i] = pre[i];
    pre[i]->reference_counter++;
  }
  return foo;
}
FGraphNode *copyGraph(const FGraphNode *node) {
  FGraphNode *foo = new FGraphNode();
  // predecessors
  foo->num_predecessor = node->num_predecessor;
  if (foo->num_predecessor) {
    foo->predecessors = safe_mal<FGraphNode *>(foo->num_predecessor);
    for (int i = 0; i < foo->num_predecessor; i++) {
      foo->predecessors[i] = node->predecessors[i];
      node->predecessors[i]->reference_counter++;
    }
  }

  foo->reference_counter =
      1; // is not copied since it is not yet referenced in contrast to node
  FOperation *op = new FOperation();
  foo->operation = op;
  op->data_type = node->operation->data_type;
  op->op_type = node->operation->op_type;
  op->dimensions = node->operation->dimensions;
  // shape
  if (op->dimensions) {
    op->shape = safe_mal<int>(op->dimensions);
    std::memcpy(op->shape, node->operation->shape,
                op->dimensions * sizeof(int));
  }
  // additional data
  if (node->operation->additional_data) {
    void **data = nullptr;
    void *src = nullptr;
    int num_entries = 0;
    switch (op->op_type) {
    case RESULTDATA: {
      FResultData *ord = (FResultData *)node->operation->additional_data;
      FResultData *crd = new FResultData();
      op->additional_data = (void *)crd;
      crd->mem_id = nullptr;
      crd->num_entries = ord->num_entries;
      num_entries = crd->num_entries;
      src = ord->data;
      data = &crd->data;
    } break;
    case STORE: {
      FStore *ord = (FStore *)node->operation->additional_data;
      FStore *crd = new FStore();
      op->additional_data = (void *)crd;
      crd->mem_id = nullptr;
      crd->num_entries = ord->num_entries;
      num_entries = crd->num_entries;
      src = ord->data;
      data = &crd->data;
    } break;
    case CONST: {
      FConst *ord = (FConst *)node->operation->additional_data;
      FConst *crd = new FConst();
      op->additional_data = (void *)crd;
      num_entries = 1;
      src = ord->value;
      data = &crd->value;
    } break;
    default:
      break;
    }
    if (data) {
      size_t byte_size = num_entries;
      switch (op->data_type) {
      case INT32:
        *data = safe_mal<int>(num_entries);
        byte_size *= sizeof(int);
        break;
      case INT64:
        *data = safe_mal<long>(num_entries);
        byte_size *= sizeof(long);
        break;
      case FLOAT32:
        *data = safe_mal<float>(num_entries);
        byte_size *= sizeof(float);
        break;
      case FLOAT64:
        *data = safe_mal<double>(num_entries);
        byte_size *= sizeof(double);
        break;
      }
      std::memcpy(*data, src, byte_size);
    }
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
  op->op_type = POW;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
FGraphNode *pow(FGraphNode *a, double b) { return pow<double>(a, b); }
FGraphNode *pow(FGraphNode *a, float b) { return pow<float>(a, b); }
FGraphNode *pow(FGraphNode *a, int b) { return pow<int>(a, b); }
FGraphNode *pow(FGraphNode *a, long b) { return pow<long>(a, b); }

FGraphNode *flatten(FGraphNode *a) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FLATTEN;
  op->dimensions = 1;
  op->shape = safe_mal<int>(1);
  const FOperation *prev_op = a->operation;
  size_t total_size = 1;
  for (int i = 0; i < prev_op->dimensions; i++)
    total_size *= prev_op->shape[i];
  op->shape[0] = total_size;
  op->additional_data = prev_op->additional_data;
  op->data_type = prev_op->data_type;
  return addNode(op, {a});
}
FGraphNode *flatten(FGraphNode *a, int dimension) {
  if (dimension == 0)
    log(ERROR, "Flattening the first dimension of a tensor is not possible!");

  FOperation *prev_op = a->operation;
  int new_prevdim_size =
      prev_op->shape[dimension - 1] * prev_op->shape[dimension];
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FLATTEN;
  op->dimensions = prev_op->dimensions - 1;
  op->shape = safe_mal<int>(prev_op->dimensions - 1);
  // copy into shape
  memcpy(op->shape, prev_op->shape, sizeof(int) * dimension);
  memcpy(op->shape + dimension, prev_op->shape + (dimension + 1),
         sizeof(int) * (prev_op->dimensions - dimension - 1));
  op->shape[dimension - 1] = new_prevdim_size;

  op->additional_data = prev_op->additional_data;
  op->data_type = prev_op->data_type;
  return addNode(op, {a});
}
