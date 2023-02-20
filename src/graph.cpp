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
#include "utils.hpp"
#include <cstring>
#include <iostream>
#include <list>
#include <stdlib.h>
#include <string>
#include <unordered_set>
#include <vector>
#define MAX(x, y) (x) > (y) ? (x) : (y)
static bool use_cpu, use_gpu, eager_execution = false;
// converts c++ type to flint type

// EAGER EXECUTION WITH HELPER
void enable_eager_execution() { eager_execution = true; }
void disable_eager_execution() { eager_execution = false; }
int is_eager_execution() { return eager_execution; }
static inline FGraphNode *execute_eagerly(FGraphNode *f) {
  const FOperation *fop = f->operation;
  bool all_calculated = true;
  for (int i = 0; i < f->num_predecessor; i++) {
    if (fop->op_type != FSTORE && !f->result_data && fop->op_type != FCONST) {
      all_calculated = false;
      break;
    }
  }
  if (all_calculated) {
    // since we only have one node the heuristics become constant
    unsigned int gpu_score = 0;
    if (fop->op_type == FMATMUL) {
      FOperation *pred0 = f->predecessors[0]->operation;
      FOperation *pred1 = f->predecessors[1]->operation;
      size_t total = 0;
      for (FOperation *predop :
           {pred0, pred1}) // FStore and FResultData have the same alignment so
                           // casting is ok
        total = MAX(total, ((FStore *)predop->additional_data)->num_entries);
      gpu_score += total * pred0->shape[pred0->dimensions - 1];
    } else if (fop->op_type == FREDUCE_SUM || fop->op_type == FREDUCE_MUL) {
      FOperation *pred0 = f->predecessors[0]->operation;
      gpu_score += ((FStore *)pred0->additional_data)->num_entries *
                   pred0->shape[((int *)fop->additional_data)[0]];
    }
    return gpu_score > 2048 ? fExecuteGraph_gpu_eagerly(f)
                            : fExecuteGraph_cpu_eagerly(f);
  } else {
    return fExecuteGraph(f);
  }
}

// INTERFACE METHODS
FGraphNode *fExecuteGraph(FGraphNode *node) {
  // TODO
  if (use_gpu)
    return fExecuteGraph_gpu(node);
  if (use_cpu)
    return fExecuteGraph_cpu(node);
  return nullptr;
}
void flintCleanup() {
  flintCleanup_cpu();
  flintCleanup_gpu();
}
void flintInit(int cpu, int gpu) {
  flogging(F_VERBOSE, "Initializing Flint");
  use_cpu = cpu;
  use_gpu = gpu;
  if (cpu)
    flintInit_cpu();
  if (gpu)
    flintInit_gpu();
}
// GRAPH METHODS
FGraphNode *fCreateGraph(const void *data, const int num_entries,
                         const FType data_type, const size_t *shape,
                         const int dimensions) {
  FGraphNode *gn = new FGraphNode();
  gn->reference_counter = 0;
  gn->result_data = nullptr;
  FOperation *op = new FOperation();
  FStore *store = new FStore();
  op->dimensions = dimensions;
  op->shape = safe_mal<size_t>(dimensions);
  std::memcpy((void *)op->shape, (void *)shape, dimensions * sizeof(size_t));
  op->additional_data = (void *)store;
  op->op_type = FSTORE;
  size_t byte_size = num_entries;
  switch (data_type) {
  case F_INT32:
    store->data = safe_mal<int>(num_entries);
    byte_size *= sizeof(int);
    break;
  case F_INT64:
    store->data = safe_mal<long>(num_entries);
    byte_size *= sizeof(long);
    break;
  case F_FLOAT32:
    store->data = safe_mal<float>(num_entries);
    byte_size *= sizeof(float);
    break;
  case F_FLOAT64:
    store->data = safe_mal<long>(num_entries);
    byte_size *= sizeof(double);
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
void fFreeGraph(FGraphNode *graph) {
  std::unordered_set<FGraphNode *>
      all; // all which are in the queue and were visited
  std::list<FGraphNode *> wq;
  all.insert(graph);
  wq.push_back(graph);
  while (!wq.empty()) {
    FGraphNode *gn = wq.front();
    wq.pop_front();
    if (gn->reference_counter != 0) {
      continue;
    }
    for (int i = 0; i < gn->num_predecessor; i++) {
      if (gn->predecessors[i] &&
          --(gn->predecessors[i]->reference_counter) == 0 &&
          all.find(gn->predecessors[i]) == all.end()) {
        wq.push_back(gn->predecessors[i]);
        all.insert(gn->predecessors[i]);
      }
    }
    bool freed_res = false;
    if (gn->result_data) {
      freed_res = true;
      FResultData *rd = gn->result_data;
      if (rd->data)
        free(rd->data);
      if (rd->mem_id)
        clReleaseMemObject(rd->mem_id);
      delete rd;
    }
    if (gn->predecessors != NULL && gn->num_predecessor != 0)
      free(gn->predecessors);
    if (gn->operation != NULL) {
      if (gn->operation->shape)
        free(gn->operation->shape);
      if (gn->operation->additional_data)
        switch (gn->operation->op_type) {
        case FSTORE: {
          FStore *st = (FStore *)gn->operation->additional_data;
          if (!freed_res) {
            free(st->data);
            if (st->mem_id)
              clReleaseMemObject(st->mem_id);
          }
          delete st;
        } break;
        case FCONST: {
          FConst *c = (FConst *)gn->operation->additional_data;
          free(c->value);
          delete c;
        } break;
        case FSLICE: {
          FSlice *s = (FSlice *)gn->operation->additional_data;
          free(s->end);
          free(s->start);
          free(s->step);
          delete s;
        } break;
        case FTRANSPOSE:
        case FREDUCE_SUM:
        case FREDUCE_MUL:
          free(gn->operation->additional_data);
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
    flogging(F_WARNING,
             "You are adding a node with a NULL operation, this is not "
             "correct behaviour!");
  }
  FGraphNode *foo = new FGraphNode();
  foo->reference_counter = 0;
  foo->operation = op;
  foo->result_data = nullptr;
  foo->num_predecessor = pre.size();
  foo->predecessors =
      pre.size() == 0 ? NULL : safe_mal<FGraphNode *>(pre.size());
  for (size_t i = 0; i < pre.size(); i++) {
    foo->predecessors[i] = pre[i];
    pre[i]->reference_counter++;
  }
  return eager_execution ? execute_eagerly(foo) : foo;
}
FGraphNode *fCopyGraph(const FGraphNode *node) {
  FGraphNode *foo = new FGraphNode();
  // predecessors
  foo->result_data = nullptr;
  if (node->result_data) {
    FResultData *ord = node->result_data;
    FResultData *crd = new FResultData();
    foo->result_data = crd;
    crd->mem_id = nullptr;
    crd->num_entries = ord->num_entries;
    size_t byte_size = crd->num_entries;
    switch (node->operation->data_type) {
    case F_INT32:
      crd->data = safe_mal<int>(crd->num_entries);
      byte_size *= sizeof(int);
      break;
    case F_INT64:
      crd->data = safe_mal<long>(crd->num_entries);
      byte_size *= sizeof(long);
      break;
    case F_FLOAT32:
      crd->data = safe_mal<float>(crd->num_entries);
      byte_size *= sizeof(float);
      break;
    case F_FLOAT64:
      crd->data = safe_mal<double>(crd->num_entries);
      byte_size *= sizeof(double);
      break;
    }
    std::memcpy(crd->data, ord->data, byte_size);
  }
  foo->num_predecessor = node->num_predecessor;
  if (foo->num_predecessor) {
    foo->predecessors = safe_mal<FGraphNode *>(foo->num_predecessor);
    for (int i = 0; i < foo->num_predecessor; i++) {
      foo->predecessors[i] = node->predecessors[i];
      node->predecessors[i]->reference_counter++;
    }
  }

  foo->reference_counter =
      0; // is not copied since it is not yet referenced in contrast to node
  FOperation *op = new FOperation();
  foo->operation = op;
  op->data_type = node->operation->data_type;
  op->op_type = node->operation->op_type;
  op->dimensions = node->operation->dimensions;
  // shape
  if (op->dimensions) {
    op->shape = safe_mal<size_t>(op->dimensions);
    std::memcpy(op->shape, node->operation->shape,
                op->dimensions * sizeof(size_t));
  }
  // additional data
  if (node->operation->additional_data) {
    void **data = nullptr;
    void *src = nullptr;
    size_t num_entries = 0;
    switch (op->op_type) {
    case FSTORE: {
      FStore *ord = (FStore *)node->operation->additional_data;
      FStore *crd = new FStore();
      op->additional_data = (void *)crd;
      crd->mem_id = nullptr;
      crd->num_entries = ord->num_entries;
      num_entries = crd->num_entries;
      src = ord->data;
      data = &crd->data;
    } break;
    case FCONST: {
      FConst *ord = (FConst *)node->operation->additional_data;
      FConst *crd = new FConst();
      op->additional_data = (void *)crd;
      num_entries = 1;
      src = ord->value;
      data = &crd->value;
    } break;
    case FSLICE: {
      FSlice *osl = (FSlice *)node->operation->additional_data;
      FSlice *csl = new FSlice();
      op->additional_data = (void *)csl;
      csl->start = osl->start;
      csl->end = osl->end;
      csl->step = osl->step;
    } break;
    case FREDUCE_SUM:
    case FREDUCE_MUL: {
      op->additional_data = safe_mal<int>(1);
      ((int *)op->additional_data)[0] =
          ((int *)node->operation->additional_data)[0];
    }
    default:
      break;
    }
    if (data) {
      size_t byte_size = num_entries;
      switch (op->data_type) {
      case F_INT32:
        *data = safe_mal<int>(num_entries);
        byte_size *= sizeof(int);
        break;
      case F_INT64:
        *data = safe_mal<long>(num_entries);
        byte_size *= sizeof(long);
        break;
      case F_FLOAT32:
        *data = safe_mal<float>(num_entries);
        byte_size *= sizeof(float);
        break;
      case F_FLOAT64:
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
  size_t *src = nullptr;
  size_t *lower = nullptr;
  int lower_dim = -1;
  if (!b || a->dimensions >= b->dimensions) {
    op->dimensions = a->dimensions;
    src = a->shape;
    if (b) {
      lower = b->shape;
      lower_dim = b->dimensions;
    }
  } else {
    op->dimensions = b->dimensions;
    src = b->shape;
    lower = a->shape;
    lower_dim = a->dimensions;
  }
  // check shape if both are defined and lower is not a constant
  if (lower && !(lower_dim == 1 && lower[0] == 1)) {
    for (int i = 0; i < lower_dim; i++)
      if (src[i + (op->dimensions - lower_dim)] != lower[i])
        flogging(
            F_ERROR,
            "incompatible shapes of operands: " +
                vectorString(std::vector<size_t>(src, src + op->dimensions)) +
                " and " +
                vectorString(std::vector<size_t>(lower, lower + lower_dim)));
  }
  op->shape = (size_t *)malloc(sizeof(size_t) * op->dimensions);
  memcpy((void *)op->shape, src, sizeof(size_t) * op->dimensions);
  // determine type
  FType highest = F_INT32;
  if (a->data_type == F_FLOAT64 || (b && b->data_type == F_FLOAT64))
    highest = F_FLOAT64;
  else if (a->data_type == F_FLOAT32 || (b && b->data_type == F_FLOAT32))
    highest = F_FLOAT32;
  else if (a->data_type == F_INT64 || (b && b->data_type == F_INT64))
    highest = F_INT64;
  op->data_type = highest;
}
FGraphNode *fadd_g(FGraphNode *a, FGraphNode *b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FADD;
  initShape_keep(op, a->operation, b->operation);
  return addNode(op, {a, b});
}
FGraphNode *fsub_g(FGraphNode *a, FGraphNode *b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FSUB;
  initShape_keep(op, a->operation, b->operation);
  return addNode(op, {a, b});
}
FGraphNode *fdiv_g(FGraphNode *a, FGraphNode *b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FDIV;
  initShape_keep(op, a->operation, b->operation);
  return addNode(op, {a, b});
}
FGraphNode *fmul_g(FGraphNode *a, FGraphNode *b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FMUL;
  initShape_keep(op, a->operation, b->operation);
  return addNode(op, {a, b});
}
FGraphNode *fpow_g(FGraphNode *a, FGraphNode *b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FPOW;
  initShape_keep(op, a->operation, b->operation);
  return addNode(op, {a, b});
}
FGraphNode *fmin_g(FGraphNode *a, FGraphNode *b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FMIN;
  initShape_keep(op, a->operation, b->operation);
  return addNode(op, {a, b});
}
FGraphNode *fmax_g(FGraphNode *a, FGraphNode *b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FMAX;
  initShape_keep(op, a->operation, b->operation);
  return addNode(op, {a, b});
}
template <typename T>
static FGraphNode *addNodeWithConst(FOperation *op, FGraphNode *a, const T b) {
  FConst *cons = new FConst();
  T *cons_val = (T *)malloc(sizeof(T));
  *cons_val = b;
  cons->value = (void *)cons_val;
  FOperation *cop = new FOperation();
  cop->op_type = FCONST;
  cop->dimensions = 1;
  cop->shape = safe_mal<size_t>(1);
  cop->shape[0] = 1;
  cop->additional_data = (void *)cons;
  if (typeid(T) == typeid(int))
    cop->data_type = F_INT32;
  else if (typeid(T) == typeid(long))
    cop->data_type = F_INT64;
  else if (typeid(T) == typeid(float))
    cop->data_type = F_FLOAT32;
  else if (typeid(T) == typeid(double))
    cop->data_type = F_FLOAT64;
  return addNode(op, {a, addNode(cop, {})});
}
template <typename T>
static FGraphNode *addConstWithNode(FOperation *op, const T b, FGraphNode *a) {
  FConst *cons = new FConst();
  T *cons_val = (T *)malloc(sizeof(T));
  *cons_val = b;
  cons->value = (void *)cons_val;
  FOperation *cop = new FOperation();
  cop->op_type = FCONST;
  cop->dimensions = 1;
  cop->shape = safe_mal<size_t>(1);
  cop->shape[0] = 1;
  cop->additional_data = (void *)cons;
  if (typeid(T) == typeid(int))
    cop->data_type = F_INT32;
  else if (typeid(T) == typeid(long))
    cop->data_type = F_INT64;
  else if (typeid(T) == typeid(float))
    cop->data_type = F_FLOAT32;
  else if (typeid(T) == typeid(double))
    cop->data_type = F_FLOAT64;
  return addNode(op, {addNode(cop, {}), a});
}
// creates tensor consisting of a single value
template <typename T>
static inline FGraphNode *constant(const T value, const size_t *shape,
                                   const int dimensions) {
  FOperation *op = new FOperation();
  op->dimensions = dimensions;
  op->shape = safe_mal<size_t>(dimensions);
  memcpy(op->shape, shape, op->dimensions * sizeof(size_t));
  op->op_type = FSTORE;
  op->data_type = toFlintType<T>();
  FStore *store = new FStore();
  op->additional_data = store;
  size_t total_size = 1;
  for (int i = 0; i < dimensions; i++)
    total_size *= shape[i];
  store->data = safe_mal<T>(total_size);
  std::fill((T *)store->data, (T *)store->data + total_size, value);
  store->num_entries = total_size;
  store->mem_id = nullptr;
  return addNode(op, {});
}

FGraphNode *fconstant_i(const int value, const size_t *shape,
                        const int dimensions) {
  return constant(value, shape, dimensions);
}
FGraphNode *fconstant_l(const long value, const size_t *shape,
                        const int dimensions) {
  return constant(value, shape, dimensions);
}

FGraphNode *fconstant_f(const float value, const size_t *shape,
                        const int dimensions) {
  return constant(value, shape, dimensions);
}

FGraphNode *fconstant_d(const double value, const size_t *shape,
                        const int dimensions) {
  return constant(value, shape, dimensions);
}

// adds the constant value to each entry in a
template <typename T> static inline FGraphNode *add(FGraphNode *a, const T b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FADD;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
FGraphNode *fadd_cd(FGraphNode *a, const double b) { return add<double>(a, b); }
FGraphNode *fadd_cf(FGraphNode *a, const float b) { return add<float>(a, b); }
FGraphNode *fadd_ci(FGraphNode *a, const int b) { return add<int>(a, b); }
FGraphNode *fadd_cl(FGraphNode *a, const long b) { return add<long>(a, b); }
// subtracts the constant value from each entry in a
template <typename T> static inline FGraphNode *sub(FGraphNode *a, const T b) {
  FOperation *op = new FOperation();
  op->op_type = FSUB;
  op->additional_data = nullptr;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
template <typename T> static inline FGraphNode *sub(const T b, FGraphNode *a) {
  FOperation *op = new FOperation();
  op->op_type = FSUB;
  op->additional_data = nullptr;
  initShape_keep(op, a->operation, nullptr);
  return addConstWithNode(op, b, a);
}
FGraphNode *fsub_cd(FGraphNode *a, const double b) { return sub<double>(a, b); }
FGraphNode *fsub_cf(FGraphNode *a, const float b) { return sub<float>(a, b); }
FGraphNode *fsub_ci(FGraphNode *a, const int b) { return sub<int>(a, b); }
FGraphNode *fsub_cl(FGraphNode *a, const long b) { return sub<long>(a, b); }

FGraphNode *fsub_icd(const double b, FGraphNode *a) {
  return sub<double>(b, a);
}
FGraphNode *fsub_icf(const float b, FGraphNode *a) { return sub<float>(b, a); }
FGraphNode *fsub_ici(const int b, FGraphNode *a) { return sub<int>(b, a); }
FGraphNode *fsub_icl(const long b, FGraphNode *a) { return sub<long>(b, a); }
// divides each entry in a by the constant value
template <typename T> static inline FGraphNode *div(FGraphNode *a, const T b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FDIV;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
template <typename T> static inline FGraphNode *div(const T b, FGraphNode *a) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FDIV;
  initShape_keep(op, a->operation, nullptr);
  return addConstWithNode(op, b, a);
}
FGraphNode *fdiv_cd(FGraphNode *a, const double b) { return div<double>(a, b); }
FGraphNode *fdiv_cf(FGraphNode *a, const float b) { return div<float>(a, b); }
FGraphNode *fdiv_ci(FGraphNode *a, const int b) { return div<int>(a, b); }
FGraphNode *fdiv_cl(FGraphNode *a, const long b) { return div<long>(a, b); }

FGraphNode *fdiv_icd(const double b, FGraphNode *a) {
  return div<double>(b, a);
}
FGraphNode *fdiv_icf(const float b, FGraphNode *a) { return div<float>(b, a); }
FGraphNode *fdiv_ici(const int b, FGraphNode *a) { return div<int>(b, a); }
FGraphNode *fdiv_icl(const long b, FGraphNode *a) { return div<long>(b, a); }
// multiplicates the constant value with each entry in a
template <typename T> static inline FGraphNode *mul(FGraphNode *a, const T b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FMUL;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
FGraphNode *fmul_cd(FGraphNode *a, const double b) { return mul<double>(a, b); }
FGraphNode *fmul_cf(FGraphNode *a, const float b) { return mul<float>(a, b); }
FGraphNode *fmul_ci(FGraphNode *a, const int b) { return mul<int>(a, b); }
FGraphNode *fmul_cl(FGraphNode *a, const long b) { return mul<long>(a, b); }
// takes the power of each element in a to b
template <typename T> static inline FGraphNode *pow(FGraphNode *a, const T b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FPOW;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
FGraphNode *fpow_cd(FGraphNode *a, const double b) { return pow<double>(a, b); }
FGraphNode *fpow_cf(FGraphNode *a, const float b) { return pow<float>(a, b); }
FGraphNode *fpow_ci(FGraphNode *a, const int b) { return pow<int>(a, b); }
FGraphNode *fpow_cl(FGraphNode *a, const long b) { return pow<long>(a, b); }

template <typename T> static inline FGraphNode *min(FGraphNode *a, const T b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FMIN;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
FGraphNode *fmin_ci(FGraphNode *a, const int b) { return min(a, b); }
FGraphNode *fmin_cl(FGraphNode *a, const long b) { return min(a, b); }
FGraphNode *fmin_cf(FGraphNode *a, const float b) { return min(a, b); }
FGraphNode *fmin_cd(FGraphNode *a, const double b) { return min(a, b); }

template <typename T> static inline FGraphNode *max(FGraphNode *a, const T b) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FMAX;
  initShape_keep(op, a->operation, nullptr);
  return addNodeWithConst(op, a, b);
}
FGraphNode *fmax_ci(FGraphNode *a, const int b) { return max(a, b); }
FGraphNode *fmax_cl(FGraphNode *a, const long b) { return max(a, b); }
FGraphNode *fmax_cf(FGraphNode *a, const float b) { return max(a, b); }
FGraphNode *fmax_cd(FGraphNode *a, const double b) { return max(a, b); }

static inline FGraphNode *log_impl(FGraphNode *a,
                                   const FOperationType logtype) {
  FOperation *op = new FOperation();
  op->op_type = logtype;
  op->dimensions = a->operation->dimensions;
  op->shape = safe_mal<size_t>(op->dimensions * sizeof(size_t));
  memcpy(op->shape, a->operation->shape, op->dimensions * sizeof(size_t));
  op->data_type = a->operation->data_type;
  if (op->data_type == F_INT32 || op->data_type == F_INT64)
    op->data_type = F_FLOAT64;
  return addNode(op, {a});
}
/** Takes the elementwise natural logarithm of a */
FGraphNode *flog(FGraphNode *a) { return log_impl(a, FLOG); }
/** Takes the elementwise logarithm of a to the basis of 2*/
FGraphNode *flog2(FGraphNode *a) { return log_impl(a, FLOG2); }
/** Takes the elementwise logarithm of a to the basis of 10*/
FGraphNode *flog10(FGraphNode *a) { return log_impl(a, FLOG10); }

/** Negates the elements of the tensor */
FGraphNode *fneg(FGraphNode *a) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FNEG;
  op->dimensions = a->operation->dimensions;
  op->shape = safe_mal<size_t>(op->dimensions);
  memcpy(op->shape, a->operation->shape, op->dimensions * sizeof(size_t));
  op->data_type = a->operation->data_type;
  return addNode(op, {a});
}
FGraphNode *fsign(FGraphNode *a) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FSIGN;
  op->dimensions = a->operation->dimensions;
  op->shape = safe_mal<size_t>(op->dimensions);
  memcpy(op->shape, a->operation->shape, op->dimensions * sizeof(size_t));
  op->data_type = a->operation->data_type;
  return addNode(op, {a});
}
FGraphNode *feven(FGraphNode *a) {
  if (a->operation->data_type != F_INT32 && a->operation->data_type != F_INT64)
    flogging(F_ERROR,
             "Can't compute if tensor is even for floating point tensor!");
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FEVEN;
  op->dimensions = a->operation->dimensions;
  op->shape = safe_mal<size_t>(op->dimensions);
  memcpy(op->shape, a->operation->shape, op->dimensions * sizeof(size_t));
  op->data_type = a->operation->data_type;
  return addNode(op, {a});
}
FGraphNode *fflatten(FGraphNode *a) {
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FLATTEN;
  op->dimensions = 1;
  op->shape = safe_mal<size_t>(1);
  const FOperation *prev_op = a->operation;
  size_t total_size = 1;
  for (int i = 0; i < prev_op->dimensions; i++)
    total_size *= prev_op->shape[i];
  op->shape[0] = total_size;
  op->data_type = prev_op->data_type;
  return addNode(op, {a});
}
FGraphNode *fflatten_dimension(FGraphNode *a, const int dimension) {
  if (dimension == 0)
    flogging(F_ERROR,
             "Flattening the first dimension of a tensor is not possible!");

  FOperation *prev_op = a->operation;
  size_t new_prevdim_size =
      prev_op->shape[dimension - 1] * prev_op->shape[dimension];
  FOperation *op = new FOperation();
  op->additional_data = nullptr;
  op->op_type = FLATTEN;
  op->dimensions = prev_op->dimensions - 1;
  op->shape = safe_mal<size_t>(prev_op->dimensions - 1);
  // copy into shape
  memcpy(op->shape, prev_op->shape, sizeof(size_t) * dimension);
  memcpy(op->shape + dimension, prev_op->shape + (dimension + 1),
         sizeof(size_t) * (prev_op->dimensions - dimension - 1));
  op->shape[dimension - 1] = new_prevdim_size;

  op->additional_data = nullptr;
  op->data_type = prev_op->data_type;
  return addNode(op, {a});
}

FGraphNode *fmatmul(FGraphNode *a, FGraphNode *b) {
  FGraphNode *x = a;
  FGraphNode *y = b;
  if (x->operation->op_type != FSTORE && !x->result_data) {
    x = fExecuteGraph(x);
  }
  if (y->operation->op_type != FSTORE && !y->result_data) {
    y = fExecuteGraph(y);
  }
  FOperation *ao = x->operation;
  FOperation *bo = y->operation;

  if (ao->dimensions < 2 || bo->dimensions < 2)
    flogging(
        F_ERROR,
        "Dimensions of operands of matrix multiplications must be at least 2!");
  size_t l = ao->shape[ao->dimensions - 2];
  size_t m = ao->shape[ao->dimensions - 1];
  size_t mb = bo->shape[bo->dimensions - 2];
  size_t n = bo->shape[bo->dimensions - 1];
  if (m != mb)
    flogging(
        F_ERROR,
        "Incompatible Shapes for matrix multiplications: " +
            vectorString(std::vector(ao->shape, ao->shape + ao->dimensions)) +
            " and " +
            vectorString(std::vector(bo->shape, bo->shape + bo->dimensions)));
  FOperation *res = new FOperation();
  res->dimensions = std::max(ao->dimensions, bo->dimensions);
  res->shape = safe_mal<size_t>(res->dimensions);
  if (res->dimensions > 2)
    memcpy(res->shape, (ao->dimensions >= bo->dimensions ? ao : bo)->shape,
           sizeof(size_t) * (res->dimensions - 2));
  res->shape[res->dimensions - 2] = l;
  res->shape[res->dimensions - 1] = n;
  res->data_type =
      ao->data_type > bo->data_type ? ao->data_type : bo->data_type;
  res->op_type = FMATMUL;

  FGraphNode *node = new FGraphNode();
  node->operation = res;
  node->result_data = nullptr;
  node->num_predecessor = 2;
  node->predecessors = safe_mal<FGraphNode *>(2);
  node->predecessors[0] = x;
  node->predecessors[1] = y;
  x->reference_counter++;
  y->reference_counter++;
  node->reference_counter = 0;
  return eager_execution ? execute_eagerly(node) : node;
}
FGraphNode *freshape(FGraphNode *a, size_t *newshape, int dimensions) {
  size_t total_size_node = 1;
  for (int i = 0; i < a->operation->dimensions; i++)
    total_size_node *= a->operation->shape[i];
  size_t total_size_new = 1;
  for (int i = 0; i < dimensions; i++)
    total_size_new *= newshape[i];
  if (total_size_node != total_size_new)
    flogging(F_ERROR, "To reshape a node the product of its new shape must "
                      "match the product of its old!");
  FGraphNode *node = new FGraphNode();
  node->operation = new FOperation();
  node->result_data = nullptr;
  node->operation->shape = safe_mal<size_t>(dimensions);
  std::memcpy(node->operation->shape, newshape, dimensions * sizeof(size_t));
  node->operation->data_type = a->operation->data_type;
  node->operation->op_type = FRESHAPE;
  node->operation->dimensions = dimensions;
  node->num_predecessor = 1;
  node->predecessors = safe_mal<FGraphNode *>(1);
  node->predecessors[0] = a;
  node->reference_counter = 0;
  a->reference_counter++;
  return eager_execution ? execute_eagerly(node) : node;
}
FGraphNode *fconvert(FGraphNode *a, FType newtype) {
  FGraphNode *foo = new FGraphNode();
  foo->reference_counter = 0;
  foo->num_predecessor = 1;
  foo->result_data = nullptr;
  foo->predecessors = safe_mal<FGraphNode *>(1);
  foo->predecessors[0] = a;
  a->reference_counter++;
  foo->operation = new FOperation();
  foo->operation->data_type = newtype;
  foo->operation->dimensions = a->operation->dimensions;
  foo->operation->shape = safe_mal<size_t>(a->operation->dimensions);
  memcpy(foo->operation->shape, a->operation->shape,
         sizeof(size_t) * a->operation->dimensions);
  foo->operation->op_type = FCONVERSION;
  return eager_execution ? execute_eagerly(foo) : foo;
  ;
}

static inline FGraphNode *reduce_operation(FGraphNode *x, const int dimension,
                                           FOperationType type) {
  FGraphNode *a = x;
  if (a->operation->op_type != FSTORE && !a->result_data) {
    a = fExecuteGraph(a);
  }
  FGraphNode *foo = new FGraphNode();
  foo->reference_counter = 0;
  foo->num_predecessor = 1;
  foo->result_data = nullptr;
  foo->predecessors = safe_mal<FGraphNode *>(1);
  foo->predecessors[0] = a;
  a->reference_counter++;
  FOperation *op = new FOperation();
  FOperation *other = a->operation;
  foo->operation = op;
  op->data_type = other->data_type;
  op->op_type = type;
  op->dimensions = other->dimensions - 1;
  op->shape = safe_mal<size_t>(op->dimensions);
  memcpy(op->shape, other->shape, sizeof(size_t) * dimension);
  memcpy(op->shape + dimension, other->shape + (dimension + 1),
         sizeof(size_t) * (other->dimensions - dimension - 1));
  op->additional_data = safe_mal<int>(1);
  *(int *)op->additional_data = dimension;
  return eager_execution ? execute_eagerly(foo) : foo;
}
// freduce_sum([[1,2,3], [4,5,6]], 0) = [5,7,9],
// freduce_sum([[1,2,3], [4,5,6]], 1) = [6,15]
FGraphNode *freduce_sum(FGraphNode *a, const int dimension) {
  return reduce_operation(a, dimension, FREDUCE_SUM);
}
FGraphNode *freduce_mul(FGraphNode *a, const int dimension) {
  return reduce_operation(a, dimension, FREDUCE_MUL);
}

FGraphNode *fslice_step(FGraphNode *a, const long *start, const long *end,
                        const long *step) {
  // construct nodes
  FGraphNode *foo = new FGraphNode();
  foo->num_predecessor = 1;
  foo->result_data = nullptr;
  foo->predecessors = safe_mal<FGraphNode *>(1);
  foo->predecessors[0] = a;
  foo->reference_counter = 0;
  a->reference_counter++;
  FOperation *op = new FOperation();
  foo->operation = op;
  op->op_type = FSLICE;
  op->data_type = a->operation->data_type;
  op->dimensions = a->operation->dimensions;
  op->shape = safe_mal<size_t>(op->dimensions);

  FSlice *slice = new FSlice();
  op->additional_data = (void *)slice;
  slice->step = safe_mal<long>(op->dimensions);
  slice->start = safe_mal<long>(op->dimensions);
  slice->end = safe_mal<long>(op->dimensions);
  for (size_t i = 0; i < op->dimensions; i++) {
    slice->start[i] =
        (start[i] < 0) ? (long)a->operation->shape[i] + start[i] : start[i];
    slice->end[i] =
        (end[i] < 0) ? (long)a->operation->shape[i] + end[i] : end[i];
    slice->step[i] = step[i];
    op->shape[i] = abs(slice->end[i] - slice->start[i]);
    long step_abs = abs(step[i]);
    // start element is always present
    if (op->shape[i] % step_abs == 0)
      op->shape[i] = op->shape[i] / step_abs;
    else
      op->shape[i] = op->shape[i] / step_abs + 1;
    if (op->shape[i] > a->operation->shape[i])
      flogging(F_ERROR, "Invalid slice: dimension " + std::to_string(i) +
                            " larger then target tensor! (" +
                            std::to_string(op->shape[i]) + " > " +
                            std::to_string(a->operation->shape[i]) + ")");
    if ((step[i] < 0 && (slice->end[i] > slice->start[i])) ||
        (step[i] > 0 && (slice->end[i] < slice->start[i]))) {
      flogging(F_ERROR,
               "invalid slice: combination of step sign and start and end "
               "in dimension " +
                   std::to_string(i) + " will yield empty tensor!");
    }
  }
  return eager_execution ? execute_eagerly(foo) : foo;
}
FGraphNode *fslice(FGraphNode *a, const long *start, const long *end) {
  std::vector<long> step(a->operation->dimensions, 1);
  FGraphNode *foo = fslice_step(a, start, end, &step[0]);
  return foo;
}
FGraphNode *fabs_g(FGraphNode *a) {
  FOperation *op = new FOperation();
  op->op_type = FABS;
  op->additional_data = nullptr;
  initShape_keep(op, a->operation, nullptr);
  return addNode(op, {a});
}
FGraphNode *frepeat(FGraphNode *a, int *repetitions) {
  FOperation *op = new FOperation();
  op->op_type = FREPEAT;
  op->data_type = a->operation->data_type;
  op->dimensions = a->operation->dimensions;
  op->shape = safe_mal<size_t>(op->dimensions);
  for (int dim = 0; dim < op->dimensions; dim++) {
    op->shape[dim] = a->operation->shape[dim] * (repetitions[dim] + 1);
  }
  return addNode(op, {a});
}
FGraphNode *ftranspose(FGraphNode *a, int *transpositions) {
  FOperation *op = new FOperation();
  op->op_type = FTRANSPOSE;
  op->data_type = a->operation->data_type;
  op->dimensions = a->operation->dimensions;
  op->shape = safe_mal<size_t>(op->dimensions);
  for (int i = 0; i < op->dimensions; i++) {
    op->shape[i] = a->operation->shape[transpositions[i]];
    // check that transpositions is reflexive
    if (transpositions[transpositions[i]] != i)
      flogging(
          F_ERROR,
          "Transpositions Array must be reflexive i.e for an dimension i let j "
          "be transpositions[i]. Then i = transpositions[j] must hold.");
  }
  op->additional_data = safe_mal<int>(op->dimensions);
  memcpy(op->additional_data, transpositions,
         sizeof(int) * a->operation->dimensions);
  return addNode(op, {a});
}
