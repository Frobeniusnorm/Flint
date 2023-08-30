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
#include "backend_ocl/comp.hpp"
#include "utils.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <list>
#include <stdlib.h>
#include <string>
#include <unordered_set>
#include <vector>
#define MAX(x, y) (x) > (y) ? (x) : (y)
#define ABS(x) (x) < 0 ? -(x) : (x)
const char *fop_to_string[] = {
    "FSTORE",      "FGEN_RANDOM", "FGEN_CONST",
    "FGEN_ARANGE", "FADD",        "FSUB",
    "FMUL",        "FDIV",        "FPOW",
    "FNEG",        "FLOG",        "FSIGN",
    "FEVEN",       "FLOG2",       "FLOG10",
    "FSIN",        "FCOS",        "FTAN",
    "FASIN",       "FACOS",       "FATAN",
    "FSQRT",       "FEXP",        "FLATTEN",
    "FMATMUL",     "FCONVERSION", "FRESHAPE",
    "FMIN",        "FMAX",        "FREDUCE_SUM",
    "FREDUCE_MUL", "FREDUCE_MIN", "FREDUCE_MAX",
    "FSLICE",      "FABS",        "FREPEAT",
    "FTRANSPOSE",  "FEXTEND",     "FCONCAT",
    "FLESS",       "FEQUAL",      "FGREATER",
    "FCONVOLVE",   "FSLIDE",      "FGRADIENT_CONVOLVE",
    "FINDEX",      "FSET_INDEX",  "FSLIDING_WINDOW"};
static bool use_cpu, use_gpu, eager_execution = false, gradient_context = false;
// converts c++ type to flint type
// TODO do execution of parents where necessary in parallel
// EAGER EXECUTION WITH HELPER
void fEnableEagerExecution() { eager_execution = true; }
void fDisableEagerExecution() { eager_execution = false; }
int fIsEagerExecution() { return eager_execution; }
void fStartGradientContext() { gradient_context = true; }
void fStopGradientContext() { gradient_context = false; }
bool fIsGradientContext() { return gradient_context; }
static inline FGraphNode *execute_eagerly(FGraphNode *f) {
  if (!use_cpu && !use_gpu)
    flintInit(FLINT_BACKEND_BOTH);
  bool all_calculated = true;
  for (int i = 0; i < f->num_predecessor; i++) {
    if (f->predecessors[i]->operation.op_type != FSTORE &&
        !f->predecessors[i]->result_data) {
      all_calculated = false;
      break;
    }
  }
  if (all_calculated && (use_cpu || use_gpu)) {
    // since we only have one node the heuristics become constant
    unsigned int gpu_score = computeScore(f, false);
    return use_gpu && (gpu_score >= 2048 || !use_cpu)
               ? fExecuteGraph_gpu_eagerly(f)
               : fExecuteGraph_cpu_eagerly(f);
  } else {
    if (use_gpu && use_cpu) {
      unsigned int gpu_score = computeScore(f, true);
      return gpu_score >= 2048 || !use_cpu ? fExecuteGraph_gpu(f)
                                           : fExecuteGraph_cpu(f);
    }
    if (use_gpu)
      return fExecuteGraph_gpu(f);
    else
      return fExecuteGraph_cpu(f);
  }
}
static inline void
configureGradientInformation(FGraphNode *g, std::vector<FGraphNode *> pred) {
  if (!gradient_context)
    return;
  std::unordered_set<const FGraphNode *> *gd = nullptr;
  for (FGraphNode *p : pred) {
    if (p->gradient_data) {
      if (!gd)
        gd = new std::unordered_set<const FGraphNode *>();
      std::unordered_set<const FGraphNode *> *other =
          (std::unordered_set<const FGraphNode *> *)p->gradient_data;
      gd->reserve(other->size() + gd->size());
      for (const FGraphNode *g : *other) {
        // check if it is still a variable
        if (g->gradient_data)
          gd->insert(g);
      }
    }
  }
  g->gradient_data = (void *)gd;
}

// INTERFACE METHODS
FGraphNode *fExecuteGraph(FGraphNode *node) {
  if (!use_cpu && !use_gpu)
    flintInit(FLINT_BACKEND_BOTH);
  if (eager_execution)
    return execute_eagerly(node);
  if (use_gpu && use_cpu) {
    unsigned int gpu_score = computeScore(node, true);
    return gpu_score >= 1024 ? fExecuteGraph_gpu(node)
                             : fExecuteGraph_cpu(node);
  }
  if (use_gpu)
    return fExecuteGraph_gpu(node);
  if (use_cpu)
    return fExecuteGraph_cpu(node);
  return nullptr;
}
FGraphNode *fCalculateResult(FGraphNode *node) {
  node = fExecuteGraph(node);
  fSyncMemory(node);
  return node;
}
void flintCleanup() {
  flintCleanup_cpu();
  flintCleanup_gpu();
}
void flintInit(int backends) {
  flogging(F_VERBOSE, "Initializing Flint");
  std::srand((unsigned int)(std::time(nullptr)));
  use_cpu = (backends & FLINT_BACKEND_ONLY_CPU);
  use_gpu = (backends & FLINT_BACKEND_ONLY_GPU);
  if (use_cpu)
    flintInit_cpu();
  if (use_gpu)
    flintInit_gpu();
}
int flintInitializedBackends() {
  int backends = 0;
  if (use_cpu)
    backends |= FLINT_BACKEND_ONLY_CPU;
  if (use_gpu)
    backends |= FLINT_BACKEND_ONLY_GPU;
  return backends;
}
// GRAPH METHODS
FGraphNode *fCreateGraph(const void *data, const int num_entries,
                         const FType data_type, const size_t *shape,
                         const int dimensions) {
  FGraphNode *gn = new FGraphNode();
  gn->gradient_data = nullptr;
  gn->reference_counter = 0;
  gn->result_data = nullptr;
  FOperation op;
  FStore *store = new FStore();
  store->mem_id = nullptr;
  op.dimensions = dimensions;
  op.shape = safe_mal<size_t>(dimensions);
  std::memcpy((void *)op.shape, (void *)shape, dimensions * sizeof(size_t));
  op.additional_data = (void *)store;
  op.op_type = FSTORE;
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
  op.data_type = data_type;
  gn->operation = op;
  gn->num_predecessor = 0;
  gn->predecessors = NULL;
  return gn;
}
// frees all allocated data from the graph and the nodes that are reachable
void fFreeGraph(FGraphNode *graph) {
  std::unordered_set<const FGraphNode *>
      all; // all which are in the queue and were visited
  std::list<FGraphNode *> wq;
  all.insert(graph);
  wq.push_back(graph);
  OCLCompilerThread::memory_barrier();
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
    if (gn->gradient_data) {
      delete (std::unordered_set<const FGraphNode *> *)gn->gradient_data;
    }
    bool freed_res = false;
    if (gn->result_data) {
      freed_res = true;
      FResultData *rd = gn->result_data;
      if (rd->data)
        free(rd->data);
      if (rd->mem_id)
        clReleaseMemObject(rd->mem_id);
      rd->mem_id = nullptr;
      delete rd;
    }
    if (gn->predecessors != NULL && gn->num_predecessor != 0)
      free(gn->predecessors);
    if (gn->operation.shape)
      free(gn->operation.shape);
    if (gn->operation.additional_data) {
      switch (gn->operation.op_type) {
      case FSTORE: {
        FStore *st = (FStore *)gn->operation.additional_data;
        if (!freed_res) {
          free(st->data);
          if (st->mem_id) {
            clReleaseMemObject(st->mem_id);
            st->mem_id = nullptr;
          }
        }
        delete st;
      } break;
      default:
        freeAdditionalData(gn);
      }
      gn->operation.additional_data = nullptr;
    }
    delete gn;
  }
}
// function to add nodes to the graph i.e. operations
static FGraphNode *addNode(FOperation op, std::vector<FGraphNode *> pre) {
  FGraphNode *foo = new FGraphNode();
  configureGradientInformation(foo, pre);
  foo->reference_counter = 0;
  foo->operation = op;
  foo->result_data = nullptr;
  foo->num_predecessor = pre.size();
  foo->predecessors =
      pre.size() == 0 ? NULL : safe_mal<FGraphNode *>(pre.size());
  for (size_t i = 0; i < pre.size(); i++) {
    foo->predecessors[i] = pre[i];
    if (pre[i]->reference_counter++ > 2 && !eager_execution)
      fExecuteGraph(pre[i]);
  }
  return eager_execution ? execute_eagerly(foo) : foo;
}
FGraphNode *fCopyGraph(FGraphNode *node) {
  FGraphNode *foo = new FGraphNode();
  fSyncMemory(node);
  // predecessors
  foo->result_data = nullptr;
  if (node->result_data) {
    FResultData *ord = node->result_data;
    FResultData *crd = new FResultData();
    foo->result_data = crd;
    crd->data = nullptr;
    crd->mem_id = nullptr;
    crd->num_entries = ord->num_entries;
    if (!ord->data) {
      if (!ord->mem_id)
        flogging(F_ERROR, "Result Data has no result data!");
      crd->mem_id = OCLCompilerThread::copy_memory(
          ord->mem_id, ord->num_entries * typeSize(node->operation.data_type),
          CL_MEM_READ_ONLY);
    } else {
      size_t byte_size = crd->num_entries;
      switch (node->operation.data_type) {
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
  }
  if (node->gradient_data) {
    std::unordered_set<const FGraphNode *> *other =
        (std::unordered_set<const FGraphNode *> *)node->gradient_data;
    foo->gradient_data =
        (void *)(new std::unordered_set<const FGraphNode *>(*other));
  }
  foo->num_predecessor = node->num_predecessor;
  if (foo->num_predecessor) {
    foo->predecessors = safe_mal<FGraphNode *>(foo->num_predecessor);
    for (int i = 0; i < foo->num_predecessor; i++) {
      foo->predecessors[i] = node->predecessors[i];
      if (node->predecessors[i]->reference_counter++ > 2 && !eager_execution)
        fExecuteGraph(node->predecessors[i]);
    }
  }

  foo->reference_counter =
      0; // is not copied since it is not yet referenced in contrast to node
  FOperation op;
  op.data_type = node->operation.data_type;
  op.op_type = node->operation.op_type;
  op.dimensions = node->operation.dimensions;
  // shape
  if (op.dimensions) {
    op.shape = safe_mal<size_t>(op.dimensions);
    std::memcpy(op.shape, node->operation.shape,
                op.dimensions * sizeof(size_t));
  }
  // additional data
  if (node->operation.additional_data) {
    void **data = nullptr;
    void *src = nullptr;
    size_t num_entries = 0;
    switch (op.op_type) {
    case FSTORE: {
      FStore *ord = (FStore *)node->operation.additional_data;
      FStore *crd = new FStore();
      op.additional_data = (void *)crd;
      crd->mem_id = nullptr;
      crd->num_entries = ord->num_entries;
      num_entries = crd->num_entries;
      if (foo->result_data) {
        crd->data = foo->result_data->data;
        crd->mem_id = foo->result_data->mem_id;
      } else {
        src = ord->data;
        data = &crd->data;
      }
    } break;
    case FSLICE: {
      FSlice *osl = (FSlice *)node->operation.additional_data;
      FSlice *csl = new FSlice();
      op.additional_data = (void *)csl;
      csl->start = safe_mal<long>(node->operation.dimensions);
      std::memcpy(csl->start, osl->start,
                  node->operation.dimensions * sizeof(long));
      csl->end = safe_mal<long>(node->operation.dimensions);
      std::memcpy(csl->end, osl->end,
                  node->operation.dimensions * sizeof(long));
      csl->step = safe_mal<long>(node->operation.dimensions);
      std::memcpy(csl->step, osl->step,
                  node->operation.dimensions * sizeof(long));
    } break;
    case FREDUCE_MAX:
    case FREDUCE_MIN:
    case FCONCAT:
    case FREDUCE_SUM:
    case FREDUCE_MUL: {
      op.additional_data = safe_mal<int>(1);
      ((int *)op.additional_data)[0] =
          ((int *)node->operation.additional_data)[0];
    } break;
    case FGRADIENT_CONVOLVE:
    case FSLIDE: {
      op.additional_data = safe_mal<unsigned int>(op.dimensions - 1);
      memcpy(op.additional_data, node->operation.additional_data,
             (op.dimensions - 1) * sizeof(unsigned int));
    } break;
    case FCONVOLVE: {
      op.additional_data = safe_mal<unsigned int>(op.dimensions);
      memcpy(op.additional_data, node->operation.additional_data,
             op.dimensions * sizeof(unsigned int));
    } break;
    default:
      break;
    }
    if (data) {
      size_t byte_size = num_entries;
      switch (op.data_type) {
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
  foo->operation = op;
  return foo;
}
static inline void initShape_keep(FOperation &op, const FOperation *a,
                                  const FOperation *b) {
  size_t *src = nullptr;
  size_t *lower = nullptr;
  int lower_dim = -1;
  if (!b || a->dimensions >= b->dimensions) {
    op.dimensions = a->dimensions;
    src = a->shape;
    if (b) {
      lower = b->shape;
      lower_dim = b->dimensions;
      if (a->dimensions == b->dimensions && src[0] == 1) {
        lower = a->shape;
        src = b->shape;
      }
    }
  } else {
    op.dimensions = b->dimensions;
    src = b->shape;
    lower = a->shape;
    lower_dim = a->dimensions;
  }
  // check shape if both are defined and lower is not a constant
  if (lower && !(lower_dim == 1 && lower[0] == 1)) {
    for (int i = 0; i < lower_dim; i++) {
      const size_t s1 = src[i + (op.dimensions - lower_dim)];
      const size_t s2 = lower[i];
      if (s1 != s2)
        flogging(
            F_ERROR,
            "incompatible shapes of operands: " +
                vectorString(std::vector<size_t>(src, src + op.dimensions)) +
                " and " +
                vectorString(std::vector<size_t>(lower, lower + lower_dim)) + " in " + fop_to_string[op.op_type]);
    }
  }
  op.shape = (size_t *)malloc(sizeof(size_t) * op.dimensions);
  memcpy((void *)op.shape, src, sizeof(size_t) * op.dimensions);
  // determine type
  op.data_type = b ? higherType(a->data_type, b->data_type) : a->data_type;
}
void fMarkGradientVariable(FGraphNode *node) {
  std::unordered_set<const FGraphNode *> *trace =
      node->gradient_data
          ? (std::unordered_set<const FGraphNode *> *)node->gradient_data
          : new std::unordered_set<const FGraphNode *>();
  if (node->gradient_data && trace->contains(node))
    return;
  trace->insert(node);
  node->gradient_data = (void *)trace;
}
void fUnmarkGradientVariable(FGraphNode *node) {
  if (node->gradient_data) {
    std::unordered_set<const FGraphNode *> *gd =
        (std::unordered_set<const FGraphNode *> *)node->gradient_data;
    gd->erase(node);
    if (gd->empty()) {
      delete gd;
      node->gradient_data = nullptr;
    }
  }
}
FGraphNode *fOptimizeMemory(FGraphNode *node) {
  if (!node->gradient_data && node->operation.op_type != FSTORE &&
      node->result_data) {
    FResultData *rd = node->result_data;
    // we can modify this node to a STORE operation
    freeAdditionalData(node);
    node->operation.op_type = FSTORE;
    if (flintInitializedBackends() & FLINT_BACKEND_ONLY_GPU)
      // we can do this only when all operations have been finished
      OCLCompilerThread::memory_barrier();
    for (int i = 0; i < node->num_predecessor; i++) {
      if (--node->predecessors[i]->reference_counter == 0)
        fFreeGraph(node->predecessors[i]);
    }
    node->num_predecessor = 0;
    free(node->predecessors);
    node->predecessors = nullptr;
    FStore *store = new FStore();
    store->data = rd->data;
    store->mem_id = rd->mem_id;
    store->num_entries = rd->num_entries;
    node->operation.additional_data = store;
  }
  return node;
}
FGraphNode *fadd_g(FGraphNode *a, FGraphNode *b) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FADD;
  initShape_keep(op, &a->operation, &b->operation);
  op.data_type = higherType(a->operation.data_type, b->operation.data_type);
  return addNode(op, {a, b});
}
FGraphNode *fsub_g(FGraphNode *a, FGraphNode *b) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FSUB;
  initShape_keep(op, &a->operation, &b->operation);
  op.data_type = higherType(a->operation.data_type, b->operation.data_type);
  return addNode(op, {a, b});
}
FGraphNode *fdiv_g(FGraphNode *a, FGraphNode *b) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FDIV;
  initShape_keep(op, &a->operation, &b->operation);
  op.data_type = higherType(a->operation.data_type, b->operation.data_type);
  return addNode(op, {a, b});
}
FGraphNode *fmul_g(FGraphNode *a, FGraphNode *b) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FMUL;
  initShape_keep(op, &a->operation, &b->operation);
  op.data_type = higherType(a->operation.data_type, b->operation.data_type);
  return addNode(op, {a, b});
}
FGraphNode *fpow_g(FGraphNode *a, FGraphNode *b) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FPOW;
  initShape_keep(op, &a->operation, &b->operation);
  op.data_type = higherType(a->operation.data_type, b->operation.data_type);
  return addNode(op, {a, b});
}
FGraphNode *fmin_g(FGraphNode *a, FGraphNode *b) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FMIN;
  initShape_keep(op, &a->operation, &b->operation);
  op.data_type = higherType(a->operation.data_type, b->operation.data_type);
  return addNode(op, {a, b});
}
FGraphNode *fmax_g(FGraphNode *a, FGraphNode *b) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FMAX;
  initShape_keep(op, &a->operation, &b->operation);
  op.data_type = higherType(a->operation.data_type, b->operation.data_type);
  return addNode(op, {a, b});
}
template <typename T>
static FGraphNode *addNodeWithConst(FOperation op, FGraphNode *a, const T b) {
  FStore *store = new FStore();
  T *cons_val = (T *)malloc(sizeof(T));
  cons_val[0] = b;
  store->data = cons_val;
  store->num_entries = 1;
  store->mem_id = nullptr;
  FOperation cop;
  cop.op_type = FSTORE;
  cop.dimensions = 1;
  cop.shape = safe_mal<size_t>(1);
  cop.shape[0] = 1;
  cop.additional_data = (void *)store;
  cop.data_type = toFlintType<T>();
  return addNode(op, {a, addNode(cop, {})});
}
template <typename T>
static FGraphNode *addConstWithNode(FOperation op, const T b, FGraphNode *a) {
  FStore *store = new FStore();
  T *cons_val = (T *)malloc(sizeof(T));
  *cons_val = b;
  store->data = (void *)cons_val;
  store->num_entries = 1;
  store->mem_id = nullptr;
  FOperation cop;
  cop.op_type = FSTORE;
  cop.dimensions = 1;
  cop.shape = safe_mal<size_t>(1);
  cop.shape[0] = 1;
  cop.additional_data = (void *)store;
  if (typeid(T) == typeid(int))
    cop.data_type = F_INT32;
  else if (typeid(T) == typeid(long))
    cop.data_type = F_INT64;
  else if (typeid(T) == typeid(float))
    cop.data_type = F_FLOAT32;
  else if (typeid(T) == typeid(double))
    cop.data_type = F_FLOAT64;
  return addNode(op, {addNode(cop, {}), a});
}
// creates tensor consisting of a single value
template <typename T>
static inline FGraphNode *constant(const T value, const size_t *shape,
                                   const int dimensions) {
  FOperation op;
  op.dimensions = dimensions;
  op.shape = safe_mal<size_t>(dimensions);
  memcpy(op.shape, shape, op.dimensions * sizeof(size_t));
  op.op_type = FGEN_CONSTANT;
  op.data_type = toFlintType<T>();
  op.additional_data = safe_mal<T>(1);
  ((T *)op.additional_data)[0] = value;
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

FGraphNode *farange(const size_t *shape, const int dimensions, const int ax) {
  FOperation op;
  op.dimensions = dimensions;
  op.shape = safe_mal<size_t>(dimensions);
  memcpy(op.shape, shape, op.dimensions * sizeof(size_t));
  op.op_type = FGEN_ARANGE;
  op.data_type = F_INT64;
  op.additional_data = safe_mal<long>(1);
  ((long *)op.additional_data)[0] = ax;
  return addNode(op, {});
}
// adds the constant value to each entry in a
template <typename T> static inline FGraphNode *add(FGraphNode *a, const T b) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FADD;
  initShape_keep(op, &a->operation, nullptr);
  op.data_type = higherType(a->operation.data_type, toFlintType<T>());
  FGraphNode *foo = addNodeWithConst(op, a, b);
  return foo;
}
FGraphNode *fadd_cd(FGraphNode *a, const double b) { return add<double>(a, b); }
FGraphNode *fadd_cf(FGraphNode *a, const float b) { return add<float>(a, b); }
FGraphNode *fadd_ci(FGraphNode *a, const int b) { return add<int>(a, b); }
FGraphNode *fadd_cl(FGraphNode *a, const long b) { return add<long>(a, b); }
// subtracts the constant value from each entry in a
template <typename T> static inline FGraphNode *sub(FGraphNode *a, const T b) {
  FOperation op;
  op.op_type = FSUB;
  op.additional_data = nullptr;
  initShape_keep(op, &a->operation, nullptr);
  op.data_type = higherType(a->operation.data_type, toFlintType<T>());
  return addNodeWithConst(op, a, b);
}
template <typename T> static inline FGraphNode *sub(const T b, FGraphNode *a) {
  FOperation op;
  op.op_type = FSUB;
  op.additional_data = nullptr;
  initShape_keep(op, &a->operation, nullptr);
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
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FDIV;
  initShape_keep(op, &a->operation, nullptr);
  op.data_type = higherType(a->operation.data_type, toFlintType<T>());
  return addNodeWithConst(op, a, b);
}
template <typename T> static inline FGraphNode *div(const T b, FGraphNode *a) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FDIV;
  initShape_keep(op, &a->operation, nullptr);
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
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FMUL;
  initShape_keep(op, &a->operation, nullptr);
  op.data_type = higherType(a->operation.data_type, toFlintType<T>());
  return addNodeWithConst(op, a, b);
}
FGraphNode *fmul_cd(FGraphNode *a, const double b) { return mul<double>(a, b); }
FGraphNode *fmul_cf(FGraphNode *a, const float b) { return mul<float>(a, b); }
FGraphNode *fmul_ci(FGraphNode *a, const int b) { return mul<int>(a, b); }
FGraphNode *fmul_cl(FGraphNode *a, const long b) { return mul<long>(a, b); }
// takes the power of each element in a to b
template <typename T> static inline FGraphNode *pow(FGraphNode *a, const T b) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FPOW;
  initShape_keep(op, &a->operation, nullptr);
  op.data_type = higherType(a->operation.data_type, toFlintType<T>());
  return addNodeWithConst(op, a, b);
}
FGraphNode *fpow_cd(FGraphNode *a, const double b) { return pow<double>(a, b); }
FGraphNode *fpow_cf(FGraphNode *a, const float b) { return pow<float>(a, b); }
FGraphNode *fpow_ci(FGraphNode *a, const int b) { return pow<int>(a, b); }
FGraphNode *fpow_cl(FGraphNode *a, const long b) { return pow<long>(a, b); }

template <typename T> static inline FGraphNode *min(FGraphNode *a, const T b) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FMIN;
  initShape_keep(op, &a->operation, nullptr);
  op.data_type = higherType(a->operation.data_type, toFlintType<T>());
  return addNodeWithConst(op, a, b);
}
FGraphNode *fmin_ci(FGraphNode *a, const int b) { return min(a, b); }
FGraphNode *fmin_cl(FGraphNode *a, const long b) { return min(a, b); }
FGraphNode *fmin_cf(FGraphNode *a, const float b) { return min(a, b); }
FGraphNode *fmin_cd(FGraphNode *a, const double b) { return min(a, b); }

template <typename T> static inline FGraphNode *max(FGraphNode *a, const T b) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FMAX;
  initShape_keep(op, &a->operation, nullptr);
  op.data_type = higherType(a->operation.data_type, toFlintType<T>());
  return addNodeWithConst(op, a, b);
}
FGraphNode *fmax_ci(FGraphNode *a, const int b) { return max(a, b); }
FGraphNode *fmax_cl(FGraphNode *a, const long b) { return max(a, b); }
FGraphNode *fmax_cf(FGraphNode *a, const float b) { return max(a, b); }
FGraphNode *fmax_cd(FGraphNode *a, const double b) { return max(a, b); }

static inline FGraphNode *log_impl(FGraphNode *a,
                                   const FOperationType logtype) {
  FOperation op;
  op.op_type = logtype;
  op.dimensions = a->operation.dimensions;
  op.shape = safe_mal<size_t>(op.dimensions * sizeof(size_t));
  op.additional_data = nullptr;
  memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
  op.data_type = a->operation.data_type;
  if (op.data_type == F_INT32 || op.data_type == F_INT64) {
    a = fconvert(a, F_FLOAT64);
    op.data_type = F_FLOAT64;
  }
  return addNode(op, {a});
}
/** Takes the elementwise natural logarithm of a */
FGraphNode *flog(FGraphNode *a) { return log_impl(a, FLOG); }
/** Takes the elementwise logarithm of a to the basis of 2*/
FGraphNode *flog2(FGraphNode *a) { return log_impl(a, FLOG2); }
/** Takes the elementwise logarithm of a to the basis of 10*/
FGraphNode *flog10(FGraphNode *a) { return log_impl(a, FLOG10); }
/** Takes the elementwise sinus of a */
FGraphNode *fsin(FGraphNode *a) { return log_impl(a, FSIN); }
/** Takes the elementwise cosinus of a */
FGraphNode *fcos(FGraphNode *a) { return log_impl(a, FCOS); }
/** Takes the elementwise tangents of a */
FGraphNode *ftan(FGraphNode *a) { return log_impl(a, FTAN); }
/** Takes the elementwise inverse sinus of a */
FGraphNode *fasin(FGraphNode *a) { return log_impl(a, FASIN); }
/** Takes the elementwise inverse cosinus of a */
FGraphNode *facos(FGraphNode *a) { return log_impl(a, FACOS); }
/** Takes the elementwise inverse tangents of a */
FGraphNode *fatan(FGraphNode *a) { return log_impl(a, FATAN); }
/** Takes the elementwise square root of a */
FGraphNode *fsqrt_g(FGraphNode *a) { return log_impl(a, FSQRT); }
FGraphNode *fexp(FGraphNode *a) { return log_impl(a, FEXP); }
/** Negates the elements of the tensor */
FGraphNode *fneg(FGraphNode *a) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FNEG;
  op.dimensions = a->operation.dimensions;
  op.shape = safe_mal<size_t>(op.dimensions);
  memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
  op.data_type = a->operation.data_type;
  return addNode(op, {a});
}
FGraphNode *fsign(FGraphNode *a) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FSIGN;
  op.dimensions = a->operation.dimensions;
  op.shape = safe_mal<size_t>(op.dimensions);
  memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
  op.data_type = F_INT32;
  FGraphNode *g = addNode(op, {a});
  return g;
}
FGraphNode *feven(FGraphNode *a) {
  if (a->operation.data_type != F_INT32 && a->operation.data_type != F_INT64)
    flogging(F_ERROR,
             "Can't compute if tensor is even for floating point tensor!");
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FEVEN;
  op.dimensions = a->operation.dimensions;
  op.shape = safe_mal<size_t>(op.dimensions);
  memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
  op.data_type = F_INT32;
  FGraphNode *g = addNode(op, {a});
  return g;
}
FGraphNode *fflatten(FGraphNode *a) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FLATTEN;
  op.dimensions = 1;
  op.shape = safe_mal<size_t>(1);
  const FOperation prev_op = a->operation;
  size_t total_size = 1;
  for (int i = 0; i < prev_op.dimensions; i++)
    total_size *= prev_op.shape[i];
  op.shape[0] = total_size;
  op.data_type = prev_op.data_type;
  return addNode(op, {a});
}
FGraphNode *fflatten_dimension(FGraphNode *a, const int dimension) {
  if (dimension == 0)
    flogging(F_ERROR,
             "Flattening the first dimension of a tensor is not possible!");

  const FOperation prev_op = a->operation;
  size_t new_prevdim_size =
      prev_op.shape[dimension - 1] * prev_op.shape[dimension];
  FOperation op;
  op.op_type = FLATTEN;
  op.dimensions = prev_op.dimensions - 1;
  op.shape = safe_mal<size_t>(prev_op.dimensions - 1);
  // copy into shape
  memcpy(op.shape, prev_op.shape, sizeof(size_t) * dimension);
  memcpy(op.shape + dimension, prev_op.shape + (dimension + 1),
         sizeof(size_t) * (prev_op.dimensions - dimension - 1));
  op.shape[dimension - 1] = new_prevdim_size;

  op.additional_data = nullptr;
  op.data_type = prev_op.data_type;
  return addNode(op, {a});
}

FGraphNode *fmatmul(FGraphNode *a, FGraphNode *b) {
  FGraphNode *x = a;
  FGraphNode *y = b;
  if (!x->result_data && x->operation.op_type != FSTORE) {
    x = fExecuteGraph(x);
  }
  if (!y->result_data && y->operation.op_type != FSTORE) {
    y = fExecuteGraph(y);
  }
  const FOperation ao = x->operation;
  const FOperation bo = y->operation;

  if (ao.dimensions < 2 || bo.dimensions < 2)
    flogging(
        F_ERROR,
        "Dimensions of operands of matrix multiplications must be at least 2!");
  size_t l = ao.shape[ao.dimensions - 2];
  size_t m = ao.shape[ao.dimensions - 1];
  size_t mb = bo.shape[bo.dimensions - 2];
  size_t n = bo.shape[bo.dimensions - 1];
  if (m != mb)
    flogging(F_ERROR, "Incompatible Shapes for matrix multiplications: " +
                          vectorString(std::vector<size_t>(
                              ao.shape, ao.shape + ao.dimensions)) +
                          " and " +
                          vectorString(std::vector<size_t>(
                              bo.shape, bo.shape + bo.dimensions)));
  FOperation res;
  res.dimensions = std::max(ao.dimensions, bo.dimensions);
  res.shape = safe_mal<size_t>(res.dimensions);
  if (res.dimensions > 2)
    memcpy(res.shape, (ao.dimensions >= bo.dimensions ? ao : bo).shape,
           sizeof(size_t) * (res.dimensions - 2));
  res.shape[res.dimensions - 2] = l;
  res.shape[res.dimensions - 1] = n;
  res.data_type = ao.data_type > bo.data_type ? ao.data_type : bo.data_type;
  res.op_type = FMATMUL;
  res.additional_data = nullptr;

  FGraphNode *node = new FGraphNode();
  configureGradientInformation(node, {x, y});
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
FGraphNode *freshape(FGraphNode *a, const size_t *newshape, const int dimensions) {
  size_t total_size_node = 1;
  for (int i = 0; i < a->operation.dimensions; i++)
    total_size_node *= a->operation.shape[i];
  size_t total_size_new = 1;
  for (int i = 0; i < dimensions; i++)
    total_size_new *= newshape[i];
  if (total_size_node != total_size_new)
    flogging(F_ERROR, "To reshape a node the product of its new shape must "
                      "match the product of its old!");
  FGraphNode *node = new FGraphNode();
  configureGradientInformation(node, {a});
  node->result_data = nullptr;
  node->operation.shape = safe_mal<size_t>(dimensions);
  std::memcpy(node->operation.shape, newshape, dimensions * sizeof(size_t));
  node->operation.data_type = a->operation.data_type;
  node->operation.op_type = FRESHAPE;
  node->operation.dimensions = dimensions;
  node->num_predecessor = 1;
  node->predecessors = safe_mal<FGraphNode *>(1);
  node->predecessors[0] = a;
  node->reference_counter = 0;
  if (a->reference_counter++ > 2 && !eager_execution)
    fExecuteGraph(a);
  return eager_execution ? execute_eagerly(node) : node;
}
FGraphNode *fconvert(FGraphNode *a, FType newtype) {
  FGraphNode *foo = new FGraphNode();
  configureGradientInformation(foo, {a});
  foo->reference_counter = 0;
  foo->num_predecessor = 1;
  foo->result_data = nullptr;
  foo->predecessors = safe_mal<FGraphNode *>(1);
  foo->predecessors[0] = a;
  if (a->reference_counter++ > 2 && !eager_execution)
    fExecuteGraph(a);
  foo->operation.data_type = newtype;
  foo->operation.dimensions = a->operation.dimensions;
  foo->operation.shape = safe_mal<size_t>(a->operation.dimensions);
  memcpy(foo->operation.shape, a->operation.shape,
         sizeof(size_t) * a->operation.dimensions);
  foo->operation.op_type = FCONVERSION;
  foo->operation.additional_data = nullptr;
  return eager_execution ? execute_eagerly(foo) : foo;
  ;
}

static inline FGraphNode *reduce_operation(FGraphNode *x, const int dimension,
                                           FOperationType type) {
  FGraphNode *a = x;
  if (a->operation.op_type != FSTORE && !a->result_data) {
    a = fExecuteGraph(a);
  }
  FGraphNode *foo = new FGraphNode();
  configureGradientInformation(foo, {a});
  foo->reference_counter = 0;
  foo->num_predecessor = 1;
  foo->result_data = nullptr;
  foo->predecessors = safe_mal<FGraphNode *>(1);
  foo->predecessors[0] = a;
  if (a->reference_counter++ > 2 && !eager_execution)
    fExecuteGraph(a);
  FOperation op;
  const FOperation other = a->operation;
  op.data_type = other.data_type;
  op.op_type = type;
  if (other.dimensions > 1) {
    op.dimensions = other.dimensions - 1;
    op.shape = safe_mal<size_t>(op.dimensions);
    memcpy(op.shape, other.shape, sizeof(size_t) * dimension);
    memcpy(op.shape + dimension, other.shape + (dimension + 1),
           sizeof(size_t) * (other.dimensions - dimension - 1));
  } else {
    op.dimensions = 1;
    op.shape = safe_mal<size_t>(1);
    op.shape[0] = 1;
  }
  op.additional_data = safe_mal<int>(1);
  ((int *)op.additional_data)[0] = dimension;
  foo->operation = op;
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
FGraphNode *freduce_min(FGraphNode *a, const int dimension) {
  return reduce_operation(a, dimension, FREDUCE_MIN);
}
FGraphNode *freduce_max(FGraphNode *a, const int dimension) {
  return reduce_operation(a, dimension, FREDUCE_MAX);
}

FGraphNode *fslice_step(FGraphNode *a, const long *start, const long *end,
                        const long *step) {
  // construct nodes
  FGraphNode *foo = new FGraphNode();
  configureGradientInformation(foo, {a});
  foo->num_predecessor = 1;
  foo->result_data = nullptr;
  foo->predecessors = safe_mal<FGraphNode *>(1);
  foo->predecessors[0] = a;
  foo->reference_counter = 0;
  if (a->reference_counter++ > 2 && !eager_execution)
    fExecuteGraph(a);
  FOperation op;
  op.op_type = FSLICE;
  op.data_type = a->operation.data_type;
  op.dimensions = a->operation.dimensions;
  op.shape = safe_mal<size_t>(op.dimensions);

  FSlice *slice = new FSlice();
  op.additional_data = (void *)slice;
  slice->step = safe_mal<long>(op.dimensions);
  slice->start = safe_mal<long>(op.dimensions);
  slice->end = safe_mal<long>(op.dimensions);
  for (size_t i = 0; i < op.dimensions; i++) {
    if (step[i] == 0)
      flogging(F_ERROR, "Step may not be 0 for slicing!");
    slice->start[i] =
        (start[i] < 0) ? (long)a->operation.shape[i] + start[i] : start[i];
    slice->end[i] =
        (end[i] < 0) ? (long)a->operation.shape[i] + end[i] : end[i];
    slice->step[i] = step[i];
    op.shape[i] = ABS(slice->end[i] - slice->start[i]);
    long step_abs = ABS(step[i]);
    // start element is always present
    if (op.shape[i] % step_abs == 0)
      op.shape[i] = op.shape[i] / step_abs;
    else
      op.shape[i] = op.shape[i] / step_abs + 1;
    if (op.shape[i] > a->operation.shape[i])
      flogging(F_ERROR, "Invalid slice: dimension " + std::to_string(i) +
                            " larger then target tensor! (" +
                            std::to_string(op.shape[i]) + " > " +
                            std::to_string(a->operation.shape[i]) + ")");
    if ((step[i] < 0 && (slice->end[i] > slice->start[i])) ||
        (step[i] > 0 && (slice->end[i] < slice->start[i]))) {
      flogging(F_ERROR,
               "invalid slice: combination of step sign, start and end "
               "in dimension " +
                   std::to_string(i) + " will yield empty tensor! start: " +
                   std::to_string(slice->start[i]) +
                   ", end: " + std::to_string(slice->end[i]) +
                   ", step: " + std::to_string(slice->step[i]));
    }
  }
  foo->operation = op;
  return eager_execution ? execute_eagerly(foo) : foo;
}
FGraphNode *fslice(FGraphNode *a, const long *start, const long *end) {
  std::vector<long> step(a->operation.dimensions, 1);
  FGraphNode *foo = fslice_step(a, start, end, &step[0]);
  return foo;
}
FGraphNode *fabs_g(FGraphNode *a) {
  FOperation op;
  op.op_type = FABS;
  op.additional_data = nullptr;
  initShape_keep(op, &a->operation, nullptr);
  op.data_type = a->operation.data_type;
  return addNode(op, {a});
}
FGraphNode *frepeat(FGraphNode *a, int *repetitions) {
  FOperation op;
  op.op_type = FREPEAT;
  op.data_type = a->operation.data_type;
  op.dimensions = a->operation.dimensions;
  op.shape = safe_mal<size_t>(op.dimensions);
  for (int dim = 0; dim < op.dimensions; dim++) {
    op.shape[dim] = a->operation.shape[dim] * (repetitions[dim] + 1);
  }
  op.data_type = a->operation.data_type;
  op.additional_data = nullptr;
  return addNode(op, {a});
}
FGraphNode *ftranspose(FGraphNode *a, int *transpositions) {
  FOperation op;
  op.op_type = FTRANSPOSE;
  op.data_type = a->operation.data_type;
  op.dimensions = a->operation.dimensions;
  op.shape = safe_mal<size_t>(op.dimensions);
  for (int i = 0; i < op.dimensions; i++) {
    op.shape[i] = a->operation.shape[transpositions[i]];
    // check that transpositions is reflexive
    if (transpositions[transpositions[i]] != i)
      flogging(
          F_ERROR,
          "Transpositions Array must be reflexive i.e for an dimension i let j "
          "be transpositions[i]. Then i = transpositions[j] must hold.");
  }
  op.additional_data = safe_mal<int>(op.dimensions);
  memcpy(op.additional_data, transpositions,
         sizeof(int) * a->operation.dimensions);
  op.data_type = a->operation.data_type;
  return addNode(op, {a});
}
FGraphNode *fless_g(FGraphNode *a, FGraphNode *b) {
  FOperation op;
  op.op_type = FLESS;
  op.additional_data = nullptr;
  initShape_keep(op, &a->operation, &b->operation);
  op.data_type = F_INT32;
  FGraphNode *g = addNode(op, {a, b});
  return g;
}
FGraphNode *fgreater_g(FGraphNode *a, FGraphNode *b) {
  FOperation op;
  op.op_type = FGREATER;
  op.additional_data = nullptr;
  initShape_keep(op, &a->operation, &b->operation);
  op.data_type = F_INT32;
  FGraphNode *g = addNode(op, {a, b});
  return g;
}
FGraphNode *fequal_g(FGraphNode *a, FGraphNode *b) {
  FOperation op;
  op.op_type = FEQUAL;
  op.additional_data = nullptr;
  initShape_keep(op, &a->operation, &b->operation);
  op.data_type = F_INT32;
  FGraphNode *g = addNode(op, {a, b});
  return g;
}
template <typename T> static inline FGraphNode *less(FGraphNode *a, const T b) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FLESS;
  initShape_keep(op, &a->operation, nullptr);
  FGraphNode *g = addNodeWithConst(op, a, b);
  g->operation.data_type = F_INT32;
  return g;
}
FGraphNode *fless_ci(FGraphNode *a, const int b) { return less(a, b); }
FGraphNode *fless_cl(FGraphNode *a, const long b) { return less(a, b); }
FGraphNode *fless_cf(FGraphNode *a, const float b) { return less(a, b); }
FGraphNode *fless_cd(FGraphNode *a, const double b) { return less(a, b); }

template <typename T>
static inline FGraphNode *greater(FGraphNode *a, const T b) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FGREATER;
  initShape_keep(op, &a->operation, nullptr);
  op.data_type = F_INT32;
  FGraphNode *g = addNodeWithConst(op, a, b);
  return g;
}
FGraphNode *fgreater_ci(FGraphNode *a, const int b) { return greater(a, b); }
FGraphNode *fgreater_cl(FGraphNode *a, const long b) { return greater(a, b); }
FGraphNode *fgreater_cf(FGraphNode *a, const float b) { return greater(a, b); }
FGraphNode *fgreater_cd(FGraphNode *a, const double b) { return greater(a, b); }

template <typename T>
static inline FGraphNode *equal(FGraphNode *a, const T b) {
  FOperation op;
  op.additional_data = nullptr;
  op.op_type = FEQUAL;
  initShape_keep(op, &a->operation, nullptr);
  op.data_type = F_INT32;
  FGraphNode *g = addNodeWithConst(op, a, b);
  return g;
}
FGraphNode *fequal_ci(FGraphNode *a, const int b) { return equal(a, b); }
FGraphNode *fequal_cl(FGraphNode *a, const long b) { return equal(a, b); }
FGraphNode *fequal_cf(FGraphNode *a, const float b) { return equal(a, b); }
FGraphNode *fequal_cd(FGraphNode *a, const double b) { return equal(a, b); }
FGraphNode *fextend_step(FGraphNode *a, const size_t *new_shape,
                         const size_t *insert_at, const long *step_size) {
  // construct nodes
  FGraphNode *foo = new FGraphNode();
  configureGradientInformation(foo, {a});
  foo->num_predecessor = 1;
  foo->result_data = nullptr;
  foo->predecessors = safe_mal<FGraphNode *>(1);
  foo->predecessors[0] = a;
  foo->reference_counter = 0;
  if (a->reference_counter++ > 2 && !eager_execution)
    fExecuteGraph(a);
  // construct operation
  const int dimensions = a->operation.dimensions;
  FOperation op;
  op.op_type = FEXTEND;
  op.data_type = a->operation.data_type;
  op.dimensions = dimensions;
  op.shape = safe_mal<size_t>(dimensions);
  memcpy(op.shape, new_shape, dimensions * sizeof(size_t));
  // set the parallel score

  op.additional_data = new FExtend();
  foo->operation = op;
  FExtend &extend = *(FExtend *)op.additional_data;
  extend.start = safe_mal<size_t>(dimensions);
  extend.step = safe_mal<long>(dimensions);
  memcpy(extend.start, insert_at, dimensions * sizeof(size_t));
  memcpy(extend.step, step_size, dimensions * sizeof(long));
  return eager_execution ? execute_eagerly(foo) : foo;
}
FGraphNode *fextend(FGraphNode *a, const size_t *new_shape,
                    const size_t *insert_at) {
  const int dimensions = a->operation.dimensions;
  std::vector<long> steps(dimensions, 1);
  return fextend_step(a, new_shape, insert_at, steps.data());
}
FGraphNode *fconcat(FGraphNode *a, FGraphNode *b, const unsigned int axis) {
  FOperation op;
  op.op_type = FCONCAT;
  op.dimensions = a->operation.dimensions;
  op.shape = safe_mal<size_t>(a->operation.dimensions);
  std::memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
  op.shape[axis] = a->operation.shape[axis] + b->operation.shape[axis];
  for (int i = 0; i < op.dimensions; i++)
    if (i != axis && a->operation.shape[i] != b->operation.shape[i])
      flogging(F_ERROR,
               "Concatenations of two nodes excpects both to have the same "
               "size along every dimension except the concatenation one!");
  op.data_type = a->operation.data_type;
  op.additional_data = safe_mal<unsigned int>(1);
  ((unsigned int *)op.additional_data)[0] = axis;
  return addNode(op, {a, b});
}
FGraphNode *fexpand(FGraphNode *a, const unsigned int ax,
                    const unsigned int ax_size) {
  int n = a->operation.dimensions;
  std::vector<size_t> new_shape(n + 1);
  if (ax > 0)
    std::memcpy(new_shape.data(), a->operation.shape, sizeof(size_t) * ax);
  new_shape[ax] = 1;
  if (ax < n)
    std::memcpy(new_shape.data() + ax + 1, a->operation.shape + ax,
                sizeof(size_t) * (n - ax));
  if (ax_size == 0)
    return freshape(a, new_shape.data(), n + 1);
  std::vector<int> repet(n + 1, 0);
  repet[ax] = ax_size - 1;
  return frepeat(freshape(a, new_shape.data(), n + 1), repet.data());
}
FGraphNode *fconvolve(FGraphNode *a, FGraphNode *kernel, unsigned int *steps) {
  const FOperation ao = a->operation;
  const FOperation bo = kernel->operation;
  if (!a->result_data && ao.op_type != FSTORE) {
    fExecuteGraph(a);
  }
  if (!kernel->result_data && bo.op_type != FSTORE) {
    fExecuteGraph(kernel);
  }
  if (ao.dimensions != bo.dimensions)
    flogging(F_ERROR, "For a convolution the original Tensor and the filter "
                      "Kernel have to have to same number of dimensions!");
  if (ao.shape[ao.dimensions - 1] != bo.shape[bo.dimensions - 1])
    flogging(F_ERROR, "For a convolution the size of the last dimension of the "
                      "Tensor must match that of the kernel! " +
                          std::to_string(ao.shape[ao.dimensions - 1]) +
                          " vs. " +
                          std::to_string(bo.shape[bo.dimensions - 1]));
  std::vector<size_t> new_shape(ao.dimensions - 1);
  for (int i = 0; i < ao.dimensions - 1; i++)
    new_shape[i] = 1 + (ao.shape[i] - 1) / steps[i];
  FOperation op;
  op.dimensions = ao.dimensions - 1;
  op.shape = safe_mal<size_t>(op.dimensions);
  memcpy(op.shape, new_shape.data(), op.dimensions * sizeof(size_t));
  op.data_type = higherType(ao.data_type, bo.data_type);
  op.op_type = FCONVOLVE;
  op.additional_data = safe_mal<unsigned int>(op.dimensions);
  memcpy(op.additional_data, steps, op.dimensions * sizeof(unsigned int));
  return addNode(op, {a, kernel});
}
FGraphNode *fslide(FGraphNode *a, FGraphNode *kernel, unsigned int *steps) {
  const FOperation ao = a->operation;
  const FOperation bo = kernel->operation;
  if (!a->result_data && ao.op_type != FSTORE) {
    fExecuteGraph(a);
  }
  if (ao.dimensions != bo.dimensions)
    flogging(F_ERROR,
             "For the slide operation the original Tensor and the filter "
             "Kernel have to have to same number of dimensions!");
  if (ao.shape[ao.dimensions - 1] != bo.shape[bo.dimensions - 1])
    flogging(F_ERROR,
             "For the slide operation the size of the last dimension of the "
             "Tensor must match that of the kernel! " +
                 std::to_string(ao.shape[ao.dimensions - 1]) + " vs. " +
                 std::to_string(bo.shape[bo.dimensions - 1]));
  FOperation op;
  op.op_type = FSLIDE;
  op.data_type = higherType(ao.data_type, bo.data_type);
  op.dimensions = ao.dimensions;
  op.shape = safe_mal<size_t>(op.dimensions);
  memcpy(op.shape, bo.shape, op.dimensions * sizeof(size_t));
  op.additional_data = safe_mal<unsigned int>(op.dimensions - 1);
  memcpy(op.additional_data, steps, (op.dimensions - 1) * sizeof(unsigned int));
  return addNode(op, {a, kernel});
}
FGraphNode *frandom(const size_t *shape, const int dimensions) {
  FGraphNode *node = new FGraphNode();
  FOperation op;
  op.op_type = FGEN_RANDOM;
  op.dimensions = dimensions;
  op.shape = safe_mal<size_t>(dimensions);
  memcpy(op.shape, shape, dimensions * sizeof(size_t));
  op.data_type = F_FLOAT64;
  // Store current time in additional data
  std::chrono::duration<double, std::nano> tm =
      std::chrono::high_resolution_clock::now().time_since_epoch();
  double t = ((unsigned long)tm.count() % 1000000) / 100.0;
  op.additional_data = safe_mal<double>(1);
  ((double *)op.additional_data)[0] = t;
  node->operation = op;
  node->result_data = nullptr;
  node->predecessors = nullptr;
  node->num_predecessor = 0;
  node->gradient_data = nullptr;
  node->reference_counter = 0;
  return eager_execution ? execute_eagerly(node) : node;
}
FGraphNode *findex(FGraphNode *a, FGraphNode *indices) {
  if (indices->operation.dimensions > a->operation.dimensions)
    flogging(
        F_ERROR,
        "Invalid index Tensor dimensionality! Larger than indexed Tensor!");
  if (indices->operation.data_type != F_INT32 &&
      indices->operation.data_type != F_INT64)
    flogging(F_ERROR, "Only integer tensors may be used as indices!");
  for (int d = 0; d < indices->operation.dimensions - 1; d++)
    if (a->operation.shape[d] != indices->operation.shape[d])
      flogging(F_ERROR,
               "Invalid indices shape! Except for last dimension shape of "
               "indices Tensor has to be a prefix of the indexed Tensor!");

  FOperation op;
  op.op_type = FINDEX;
  op.dimensions = a->operation.dimensions;
  op.shape = safe_mal<size_t>(op.dimensions);
  memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
  op.shape[indices->operation.dimensions - 1] =
      indices->operation.shape[indices->operation.dimensions - 1];
  op.data_type = a->operation.data_type;
  op.additional_data = nullptr;
  return addNode(op, {a, indices});
}
FGraphNode *findex_set(FGraphNode *a, FGraphNode *b, FGraphNode *indices) {
  if (!indices->result_data && indices->operation.op_type != FSTORE)
    indices = fExecuteGraph(indices);
  if (!b->result_data && b->operation.op_type != FSTORE)
    b = fExecuteGraph(b);
  if (indices->operation.dimensions > b->operation.dimensions)
    flogging(
        F_ERROR,
        "Invalid index Tensor dimensionality! Larger than indexed Tensor!");
  if (indices->operation.data_type != F_INT32 &&
      indices->operation.data_type != F_INT64)
    flogging(F_ERROR, "Only integer tensors may be used as indices!");
  for (int d = 0; d < indices->operation.dimensions - 1; d++)
    if (b->operation.shape[d] != indices->operation.shape[d])
      flogging(F_ERROR,
               "Invalid indices shape! Except for last dimension shape of "
               "indices Tensor has to be a prefix of the indexed Tensor!");

  FOperation op;
  op.op_type = FSET_INDEX;
  op.dimensions = a->operation.dimensions;
  op.shape = safe_mal<size_t>(op.dimensions);
  memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
  op.data_type = a->operation.data_type;
  op.additional_data = nullptr;
  return addNode(op, {a, b, indices});
}
FGraphNode *fsliding_window(FGraphNode *a, const size_t *size,
                            const unsigned int *steps) {
  FOperation op;
  op.op_type = FSLIDING_WINDOW;
  op.dimensions = a->operation.dimensions + 1;
  op.data_type = a->operation.data_type;
  op.shape = safe_mal<size_t>(op.dimensions);
  op.shape[0] = 1;
  for (int i = 0; i < a->operation.dimensions; i++) {
    op.shape[i + 1] = size[i];
    // we slide a window of size size[i] with step size steps[i] along that
    // dimension
    size_t window_size = a->operation.shape[i] - size[i] + 1;
    window_size = window_size % steps[i] == 0 ? window_size / steps[i]
                                              : window_size / steps[i] + 1;
    op.shape[0] *= window_size;
  }
  FSlidingWindow *slidewin = new FSlidingWindow();
  slidewin->size = safe_mal<size_t>(a->operation.dimensions);
  slidewin->step = safe_mal<unsigned int>(a->operation.dimensions);
  memcpy(slidewin->size, size, a->operation.dimensions * sizeof(size_t));
  memcpy(slidewin->step, steps, a->operation.dimensions * sizeof(unsigned int));
  op.additional_data = (void *)(slidewin);
  return addNode(op, {a});
}
FGraphNode *fpermutate(FGraphNode *a, unsigned int ax) {
  size_t total_size;
  const long *perms = generatePermutation(a->operation.shape, ax, &total_size);
  FGraphNode *ind =
      fCreateGraph(perms, total_size, F_INT64, a->operation.shape, ax + 1);
  return findex(a, ind);
}
