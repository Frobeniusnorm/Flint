/* Copyright 2022 David Schwarzbeck

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

  This file includes the implementation of the CPU backend.
*/

#include "../flint.h"
#include "utils.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <list>
#include <queue>
#include <semaphore>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
// virtual maximum number of threads
#define MAX_PARALLELITY 4096
#define MIN_VAL(x, y) x < y ? x : y
#define MAX_VAL(x, y) x < y ? y : x
static bool initialized = false;
static std::vector<std::thread *> threads;

static void threadRoutine();
void flintInit_cpu() {
  if (!initialized) {
    initialized = true;
    int cores = std::thread::hardware_concurrency();
    if (!cores)
      cores = 8;
    flog(F_INFO, "Using " + std::to_string(cores) + " threads for CPU-backend");
    threads = std::vector<std::thread *>(cores);
    for (int i = 0; i < cores; i++)
      threads[i] = new std::thread(threadRoutine);
  }
}
struct CPUResultData {
  void *data;
  FType type;
  size_t num_entries;
  std::vector<size_t> shape;
};
template <typename T, typename A, typename B>
static void binaryExpression(T *result, A *data1, B *data2, FOperationType op,
                             size_t from, size_t size, int index_man_1,
                             int index_man_2, FGraphNode *curr) {
  switch (op) {
  case FADD:
    for (size_t i = from; i < from + size; i++)
      result[i] = data1[i % index_man_1] + data2[i % index_man_2];
    break;
  case FSUB:
    for (size_t i = from; i < from + size; i++)
      result[i] = data1[i % index_man_1] - data2[i % index_man_2];
    break;
  case FMUL:
    for (size_t i = from; i < from + size; i++)
      result[i] = data1[i % index_man_1] * data2[i % index_man_2];
    break;
  case FDIV:
    for (size_t i = from; i < from + size; i++)
      result[i] = data1[i % index_man_1] / data2[i % index_man_2];
    break;
  case FPOW:
    for (size_t i = from; i < from + size; i++)
      result[i] = pow(data1[i % index_man_1], data2[i % index_man_2]);
    break;
  case FMATMUL: {
    FGraphNode *gnp1 = curr->predecessors[0], *gnp2 = curr->predecessors[1];
    size_t l = gnp1->operation->shape[gnp1->operation->dimensions - 2];
    size_t m = gnp1->operation->shape[gnp1->operation->dimensions - 1];
    size_t n = gnp2->operation->shape[gnp2->operation->dimensions - 1];
    for (size_t index = from; index < from + size; index++) {
      result[index] = 0;
      // indices in node matrix
      size_t j = (index % (l * n)) / n;
      size_t k = (index % (l * n)) % n;

      // matrix number of predecessors
      size_t base_p1 = 0;
      if (gnp1->operation->dimensions > 2) {
        // get matrix number of index and then reproject
        base_p1 = (index / (l * n)) * (l * m);
      }
      size_t base_p2 = 0;
      if (gnp2->operation->dimensions > 2) {
        // get matrix number of index and then reproject
        base_p2 = (index / (l * n)) * (m * n);
      }
      for (size_t i = 0; i < m; i++) {
        result[index] +=
            data1[base_p1 + j * m + i] * data2[base_p2 + i * n + k];
      }
    }
  } break;
  case FMIN:
    for (size_t i = from; i < from + size; i++)
      result[i] = MIN_VAL(data1[i % index_man_1], data2[i % index_man_2]);
    break;
  case FMAX:
    for (size_t i = from; i < from + size; i++)
      result[i] = MAX_VAL(data1[i % index_man_1], data2[i % index_man_2]);
    break;
  default:
    break;
  }
}
// i hate this function more than anything else in this library (yet)
template <typename T>
static void executeNode(FGraphNode *node,
                        std::vector<CPUResultData> predecessor_data, T *result,
                        size_t from, size_t size) {
  switch (node->operation->op_type) {
  case FCONVERSION: {
    CPUResultData pred = predecessor_data[0];

    switch (pred.type) {
    case F_INT32:
      for (size_t i = from; i < from + size; i++)
        result[i] = (T)((int *)pred.data)[i];
      break;
    case F_INT64:
      for (size_t i = from; i < from + size; i++)
        result[i] = (T)((long *)pred.data)[i];
      break;
    case F_FLOAT32:
      for (size_t i = from; i < from + size; i++)
        result[i] = (T)((float *)pred.data)[i];
      break;
    case F_FLOAT64:
      for (size_t i = from; i < from + size; i++)
        result[i] = (T)((double *)pred.data)[i];
      break;
    }

  } break;
  case FREDUCE_SUM:
  case FREDUCE_MUL: {
    CPUResultData pred = predecessor_data[0];
    int dim = ((int *)node->operation->additional_data)[0];
    size_t it_dim = 1; // iteration size <=> product of all dimensions along dim
    for (size_t d = dim + 1; d < pred.shape.size(); d++)
      it_dim *= pred.shape[d];

    for (size_t i = from; i < from + size; i++) {
      // iterate through to-reduce dimension
      result[i] = node->operation->op_type == FREDUCE_SUM
                      ? 0
                      : 1; // init with neutral element
      for (size_t j = 0; j < pred.shape[dim]; j++) {
        const T curr =
            ((T *)pred.data)[(i / it_dim) * it_dim * pred.shape[dim] +
                             i % it_dim + j * it_dim];
        if (node->operation->op_type == FREDUCE_SUM)
          result[i] += curr;
        else
          result[i] *= curr;
      }
    }
  } break;
  case FRESHAPE:
  case FLATTEN: {
    CPUResultData pred = predecessor_data[0];
    for (size_t i = from; i < from + size; i++)
      result[i] = ((T *)pred.data)[i];
  } break;
  case FSLICE: {
    CPUResultData pred = predecessor_data[0];
    FSlice *slice = (FSlice *)node->operation->additional_data;
    // flattened shape data
    std::vector<size_t> acc_sizes(node->operation->dimensions);
    std::vector<size_t> acc_sizes_pred(acc_sizes.size());
    for (long d = node->operation->dimensions - 1; d >= 0; d--) {
      if (d == node->operation->dimensions - 1) {
        acc_sizes[d] = 1;
        acc_sizes_pred[d] = 1;
      } else {
        acc_sizes_pred[d] = acc_sizes_pred[d + 1] * pred.shape[d + 1];
        acc_sizes[d] = acc_sizes[d + 1] * node->operation->shape[d + 1];
      }
    }
    // calculate start and step size in flattened array
    size_t start = 0;
    std::vector<size_t> step(node->operation->dimensions);
    for (long d = 0; d < step.size(); d++) {
      start += slice->start[d] * acc_sizes_pred[d];
    }
    // calculate for each entry corresponding element
    for (size_t i = from; i < from + size; i++) {
      size_t j = start;
      for (size_t d = 0; d < step.size(); d++) {
        // get dimension index
        size_t di = (d == 0 ? i : i % acc_sizes[d - 1]) / acc_sizes[d];
        // reproject
        j += di * slice->step[d] * acc_sizes_pred[d];
      }
      result[i] = ((T *)pred.data)[j];
    }
  } break;
  case FABS: {
    CPUResultData pred = predecessor_data[0];
    for (size_t i = from; i < from + size; i++)
      result[i] = abs(((T *)pred.data)[i]);
  } break;
  default: { // binary operations
    CPUResultData p1 = predecessor_data[0], p2 = predecessor_data[1];
    size_t im1 = p1.num_entries, im2 = p2.num_entries;

    switch (p1.type) {
    case F_INT32:
      switch (p2.type) {
      case F_INT32:
        binaryExpression(result, (int *)p1.data, (int *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case F_FLOAT32:
        binaryExpression(result, (int *)p1.data, (float *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case F_FLOAT64:
        binaryExpression(result, (int *)p1.data, (double *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case F_INT64:
        binaryExpression(result, (int *)p1.data, (long *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      }
      break;
    case F_FLOAT32:
      switch (p2.type) {
      case F_INT32:
        binaryExpression(result, (float *)p1.data, (int *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case F_FLOAT32:
        binaryExpression(result, (float *)p1.data, (float *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case F_FLOAT64:
        binaryExpression(result, (float *)p1.data, (double *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case F_INT64:
        binaryExpression(result, (float *)p1.data, (long *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      }
      break;
    case F_FLOAT64:
      switch (p2.type) {
      case F_INT32:
        binaryExpression(result, (double *)p1.data, (int *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case F_FLOAT32:
        binaryExpression(result, (double *)p1.data, (float *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case F_FLOAT64:
        binaryExpression(result, (double *)p1.data, (double *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case F_INT64:
        binaryExpression(result, (double *)p1.data, (long *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      }
      break;
    case F_INT64:
      switch (p2.type) {
      case F_INT32:
        binaryExpression(result, (long *)p1.data, (int *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case F_FLOAT32:
        binaryExpression(result, (long *)p1.data, (float *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case F_FLOAT64:
        binaryExpression(result, (long *)p1.data, (double *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case F_INT64:
        binaryExpression(result, (long *)p1.data, (long *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      }
      break;
    }
    break;
  }
  }
}

static blocking_queue<
    std::tuple<FGraphNode *, std::vector<CPUResultData>, void *, size_t, size_t,
               std::counting_semaphore<MAX_PARALLELITY> *>>
    thread_queue;

void flintCleanup_cpu() {
  if (initialized) {
    flog(F_DEBUG, "Sending kill signal and poisson pills");
    initialized = false;
    for (size_t i = 0; i < threads.size(); i++)
      thread_queue.push_front({nullptr, {}, nullptr, 0, 0, nullptr});
    for (std::thread *t : threads) {
      t->join();
      delete t;
    }
  }
}

static void threadRoutine() {
  while (true) {
    auto [node, pred_data, result, from, to, sem] = thread_queue.pop_front();
    if (!node)
      break;

    switch (node->operation->data_type) {
    case F_FLOAT32:
      executeNode(node, pred_data, (float *)result, from, to);
      break;
    case F_FLOAT64:
      executeNode(node, pred_data, (double *)result, from, to);
      break;
    case F_INT32:
      executeNode(node, pred_data, (int *)result, from, to);
      break;
    case F_INT64:
      executeNode(node, pred_data, (long *)result, from, to);
      break;
    }
    sem->release();
  }
}
#define PARALLEL_EXECUTION_SIZE 1000 // for debugging
template <typename T>
inline void chooseExecutionMethod(FGraphNode *node,
                                  std::vector<CPUResultData> pred_data,
                                  T *result, size_t size) {
  auto start = std::chrono::high_resolution_clock::now();
  if (size >= PARALLEL_EXECUTION_SIZE) {
    size_t exeUnits = std::min(size, threads.size());
    size_t workSize = size / exeUnits;
    std::counting_semaphore<MAX_PARALLELITY> *sem =
        new std::counting_semaphore<MAX_PARALLELITY>(0);
    for (size_t i = 0; i < exeUnits; i++) {
      const size_t to = i == exeUnits - 1 ? size : (i + 1) * workSize;
      thread_queue.push_front(
          {node, pred_data, result, i * workSize, to - i * workSize, sem});
    }
    for (size_t i = 0; i < exeUnits; i++)
      sem->acquire();
    delete sem;
  } else {
    executeNode(node, pred_data, result, 0, size);
  }
  std::chrono::duration<double, std::milli> elapsed =
      std::chrono::high_resolution_clock::now() - start;
  flog(F_DEBUG, (size >= PARALLEL_EXECUTION_SIZE
                     ? std::string("Parallel Execution on CPU ")
                     : std::string("Sequential Execution on CPU ")) +
                    "took " + std::to_string(elapsed.count()) + "ms");
}
FGraphNode *fExecuteGraph_cpu_eagerly(FGraphNode *node) {
  if (!initialized)
    flintInit_cpu();
  bool is_data_node = node->operation->op_type == FSTORE ||
                      node->operation->op_type == FRESULTDATA ||
                      node->operation->op_type == FCONST;

  std::vector<CPUResultData> pred_data(node->num_predecessor);
  size_t total = 1;
  for (int i = 0; i < node->operation->dimensions; i++)
    total *= node->operation->shape[i];
  void *data;

  if (!is_data_node) {
    // build predecessor data
    for (int i = 0; i < node->num_predecessor; i++) {
      FGraphNode *pred = node->predecessors[i];
      if (pred->operation->op_type == FSTORE ||
          pred->operation->op_type == FRESULTDATA) {
        FStore *store = (FStore *)node->operation;
        pred_data[i].data = store->data;
        pred_data[i].num_entries = store->num_entries;
      } else { // FConst
        pred_data[i].num_entries = 1;
        pred_data[i].data = ((FConst *)node->operation)->value;
      }
      pred_data[i].type = pred->operation->data_type;
      pred_data[i].shape = std::vector<size_t>(pred->operation->shape,
                                               pred->operation->shape +
                                                   pred->operation->dimensions);
    }
    switch (node->operation->data_type) {
    case F_INT32:
      data = safe_mal<int>(total);
      chooseExecutionMethod(node, pred_data, (int *)data, total);
      break;
    case F_INT64:
      data = safe_mal<long>(total);
      chooseExecutionMethod(node, pred_data, (long *)data, total);
      break;
    case F_FLOAT32:
      data = safe_mal<float>(total);
      chooseExecutionMethod(node, pred_data, (float *)data, total);
      break;
    case F_FLOAT64:
      data = safe_mal<double>(total);
      chooseExecutionMethod(node, pred_data, (double *)data, total);
      break;
    }
  } else {
    size_t byte_size = 1;
    switch (node->operation->data_type) {
    case F_INT32:
      data = safe_mal<int>(total);
      byte_size = sizeof(int) * total;
      break;
    case F_INT64:
      data = safe_mal<long>(total);
      byte_size = sizeof(long) * total;
      break;
    case F_FLOAT32:
      data = safe_mal<float>(total);
      byte_size = sizeof(float) * total;
      break;
    case F_FLOAT64:
      data = safe_mal<double>(total);
      byte_size = sizeof(double) * total;
      break;
    }
    if (node->operation->op_type == FCONST)
      memcpy(data, ((FConst *)node->operation->additional_data)->value,
             byte_size);
    else {
      memcpy(data, ((FStore *)node->operation->additional_data)->data,
             byte_size);
    }
  }
  FResultData *rd = new FResultData();
  rd->data = data;
  rd->num_entries = total;
  rd->mem_id = nullptr;
  FOperation *op = new FOperation();
  op->dimensions = node->operation->dimensions;
  op->shape = safe_mal<size_t>(node->operation->dimensions);
  memcpy(op->shape, node->operation->shape,
         node->operation->dimensions * sizeof(size_t));
  op->data_type = node->operation->data_type;
  op->additional_data = (void *)rd;
  op->op_type = FRESULTDATA;
  FGraphNode *rn = new FGraphNode();
  rn->operation = op;
  rn->predecessors = safe_mal<FGraphNode *>(1);
  rn->predecessors[0] = node;
  node->reference_counter++;
  rn->num_predecessor = 1;
  rn->reference_counter = 0;
  return rn;
}

FGraphNode *fExecuteGraph_cpu(FGraphNode *node) {
  if (!initialized)
    flintInit_cpu();
  if (node->operation->op_type == FSTORE ||
      node->operation->op_type == FRESULTDATA)
    return node;
  // TODO parallel execution
  using namespace std;
  unordered_map<FGraphNode *, CPUResultData> results;
  unordered_set<FGraphNode *> inExecuteList;
  list<FGraphNode *> workList;  // traverse bottom up
  list<FGraphNode *> toExecute; // in top down order
  workList.push_front(node);
  // collect nodes
  while (!workList.empty()) {
    FGraphNode *curr = workList.front();
    workList.pop_front();
    for (int i = 0; i < curr->num_predecessor; i++) {
      workList.push_back(curr->predecessors[i]);
    }
    if (inExecuteList.find(curr) != inExecuteList.end())
      toExecute.remove(curr);
    inExecuteList.insert(curr);
    toExecute.push_front(curr);
  }
  // work them in correct oder
  for (FGraphNode *curr : toExecute) {
    // collect predecessor results
    vector<CPUResultData> predData(curr->num_predecessor);
    for (int i = 0; i < curr->num_predecessor; i++) {
      predData[i] = results[curr->predecessors[i]];
    }
    // calculate total size
    size_t size = 1;
    for (int j = 0; j < curr->operation->dimensions; j++)
      size *= curr->operation->shape[j];
    if (curr->operation->op_type == FSTORE ||
        curr->operation->op_type == FRESULTDATA ||
        curr->operation->op_type == FCONST) {
      CPUResultData foo;
      foo.shape =
          vector<size_t>(curr->operation->shape,
                         curr->operation->shape + curr->operation->dimensions);
      foo.type = curr->operation->data_type;
      switch (curr->operation->op_type) {
      case FSTORE: {
        FStore *store = (FStore *)curr->operation->additional_data;
        foo.num_entries = store->num_entries;
        foo.data = store->data;
      } break;
      case FRESULTDATA: {
        FResultData *store = (FResultData *)curr->operation->additional_data;
        foo.num_entries = store->num_entries;
        foo.data = store->data;
      } break;
      case FCONST: {
        FConst *cdata = (FConst *)curr->operation->additional_data;
        foo.num_entries = 1;
        foo.data = cdata->value;
      } break;
      default: // idc
        break;
      }
      results.insert({curr, foo});
    } else {
      // allocate result data and execute
      switch (curr->operation->data_type) {
      case F_INT32: {
        int *result = safe_mal<int>(size);
        chooseExecutionMethod(curr, predData, result, size);
        results.insert(
            {curr,
             {.data = (void *)result,
              .type = F_INT32,
              .num_entries = size,
              .shape = vector<size_t>(curr->operation->shape,
                                      curr->operation->shape +
                                          curr->operation->dimensions)}});
      } break;
      case F_INT64: {
        long *result = safe_mal<long>(size);
        chooseExecutionMethod(curr, predData, result, size);
        results.insert(
            {curr,
             {.data = (void *)result,
              .type = F_INT64,
              .num_entries = size,
              .shape = vector<size_t>(curr->operation->shape,
                                      curr->operation->shape +
                                          curr->operation->dimensions)}});
      } break;
      case F_FLOAT32: {
        float *result = safe_mal<float>(size);
        chooseExecutionMethod(curr, predData, result, size);
        results.insert(
            {curr,
             {.data = (void *)result,
              .type = F_FLOAT32,
              .num_entries = size,
              .shape = vector<size_t>(curr->operation->shape,
                                      curr->operation->shape +
                                          curr->operation->dimensions)}});
      } break;
      case F_FLOAT64: {
        double *result = safe_mal<double>(size);
        chooseExecutionMethod(curr, predData, result, size);
        results.insert(
            {curr,
             {.data = (void *)result,
              .type = F_FLOAT64,
              .num_entries = size,
              .shape = vector<size_t>(curr->operation->shape,
                                      curr->operation->shape +
                                          curr->operation->dimensions)}});
      } break;
      }
    }
  }
  CPUResultData final = results[node];
  // free all other data
  for (auto &[gn, rd] : results) {
    if (gn != node && gn->operation->op_type != FSTORE &&
        gn->operation->op_type != FRESULTDATA &&
        gn->operation->op_type != FCONST)
      free(rd.data);
  }
  // return data
  FResultData *rd = new FResultData();
  rd->data = final.data;
  rd->num_entries = final.num_entries;
  rd->mem_id = nullptr;
  FOperation *op = new FOperation();
  op->dimensions = node->operation->dimensions;
  op->shape = safe_mal<size_t>(final.shape.size());
  memcpy(op->shape, final.shape.data(), final.shape.size() * sizeof(size_t));
  op->data_type = node->operation->data_type;
  op->additional_data = (void *)rd;
  op->op_type = FRESULTDATA;
  FGraphNode *rn = new FGraphNode();
  rn->operation = op;
  rn->predecessors = safe_mal<FGraphNode *>(1);
  rn->predecessors[0] = node;
  node->reference_counter++;
  rn->num_predecessor = 1;
  rn->reference_counter = 0;
  return rn;
}
