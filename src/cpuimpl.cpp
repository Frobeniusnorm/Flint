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
#include "logger.hpp"
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

static bool initialized = false;
static std::vector<std::thread *> threads;

static void threadRoutine();
void flintInit_cpu() {
  if (!initialized) {
    initialized = true;
    int cores = std::thread::hardware_concurrency();
    if (!cores)
      cores = 8;
    log(INFO, "Using " + std::to_string(cores) + " threads for CPU-backend");
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
  case ADD:
    for (size_t i = from; i < from + size; i++)
      result[i] = data1[i % index_man_1] + data2[i % index_man_2];
    break;
  case SUB:
    for (size_t i = from; i < from + size; i++)
      result[i] = data1[i % index_man_1] - data2[i % index_man_2];
    break;
  case MUL:
    for (size_t i = from; i < from + size; i++)
      result[i] = data1[i % index_man_1] * data2[i % index_man_2];
    break;
  case DIV:
    for (size_t i = from; i < from + size; i++)
      result[i] = data1[i % index_man_1] / data2[i % index_man_2];
    break;
  case POW:
    for (size_t i = from; i < from + size; i++)
      result[i] = pow(data1[i % index_man_1], data2[i % index_man_2]);
    break;
  case MATMUL: {
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
  case STORE: {
    FStore *store = (FStore *)node->operation->additional_data;
    for (size_t i = from; i < from + size; i++)
      result[i] = ((T *)store->data)[i];
  } break;
  case RESULTDATA: {
    FResultData *store = (FResultData *)node->operation->additional_data;
    for (size_t i = from; i < from + size; i++)
      result[i] = ((T *)store->data)[i];
  } break;
  case CONST: {
    FConst *cons = (FConst *)node->operation->additional_data;
    result[from] = ((T *)cons->value)[0];
  } break;
  case FLATTEN: {
    CPUResultData pred = predecessor_data[0];
    for (size_t i = from; i < from + size; i++)
      result[i] = ((T *)pred.data)[i];
  } break;
  default: { // binary operations
    CPUResultData p1 = predecessor_data[0], p2 = predecessor_data[1];
    size_t im1 = p1.num_entries, im2 = p2.num_entries;

    switch (p1.type) {
    case INT32:
      switch (p2.type) {
      case INT32:
        binaryExpression(result, (int *)p1.data, (int *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case FLOAT32:
        binaryExpression(result, (int *)p1.data, (float *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case FLOAT64:
        binaryExpression(result, (int *)p1.data, (double *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case INT64:
        binaryExpression(result, (int *)p1.data, (long *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      }
      break;
    case FLOAT32:
      switch (p2.type) {
      case INT32:
        binaryExpression(result, (float *)p1.data, (int *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case FLOAT32:
        binaryExpression(result, (float *)p1.data, (float *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case FLOAT64:
        binaryExpression(result, (float *)p1.data, (double *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case INT64:
        binaryExpression(result, (float *)p1.data, (long *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      }
      break;
    case FLOAT64:
      switch (p2.type) {
      case INT32:
        binaryExpression(result, (double *)p1.data, (int *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case FLOAT32:
        binaryExpression(result, (double *)p1.data, (float *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case FLOAT64:
        binaryExpression(result, (double *)p1.data, (double *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case INT64:
        binaryExpression(result, (double *)p1.data, (long *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      }
      break;
    case INT64:
      switch (p2.type) {
      case INT32:
        binaryExpression(result, (long *)p1.data, (int *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case FLOAT32:
        binaryExpression(result, (long *)p1.data, (float *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case FLOAT64:
        binaryExpression(result, (long *)p1.data, (double *)p2.data,
                         node->operation->op_type, from, size, im1, im2, node);
        break;
      case INT64:
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
    log(DEBUG, "Sending kill signal and poisson pills");
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
    case FLOAT32:
      executeNode(node, pred_data, (float *)result, from, to);
      break;
    case FLOAT64:
      executeNode(node, pred_data, (double *)result, from, to);
      break;
    case INT32:
      executeNode(node, pred_data, (int *)result, from, to);
      break;
    case INT64:
      executeNode(node, pred_data, (long *)result, from, to);
      break;
    }
    sem->release();
  }
}
#define PARALLEL_EXECUTION_SIZE 1 // for debugging
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
  log(DEBUG, (size >= PARALLEL_EXECUTION_SIZE
                  ? std::string("Parallel Execution on CPU ")
                  : std::string("Sequential Execution on CPU ")) +
                 "took " + std::to_string(elapsed.count()) + "ms");
}
FGraphNode *executeGraph_cpu(FGraphNode *node) {
  if (!initialized)
    flintInit_cpu();
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
    // allocate result data and execute
    switch (curr->operation->data_type) {
    case INT32: {
      int *result = safe_mal<int>(size);
      chooseExecutionMethod(curr, predData, result, size);
      results.insert(
          {curr,
           {.data = (void *)result,
            .type = INT32,
            .num_entries = size,
            .shape = vector<size_t>(curr->operation->shape,
                                    curr->operation->shape +
                                        curr->operation->dimensions)}});
    } break;
    case INT64: {
      long *result = safe_mal<long>(size);
      chooseExecutionMethod(curr, predData, result, size);
      results.insert(
          {curr,
           {.data = (void *)result,
            .type = INT64,
            .num_entries = size,
            .shape = vector<size_t>(curr->operation->shape,
                                    curr->operation->shape +
                                        curr->operation->dimensions)}});
    } break;
    case FLOAT32: {
      float *result = safe_mal<float>(size);
      chooseExecutionMethod(curr, predData, result, size);
      results.insert(
          {curr,
           {.data = (void *)result,
            .type = FLOAT32,
            .num_entries = size,
            .shape = vector<size_t>(curr->operation->shape,
                                    curr->operation->shape +
                                        curr->operation->dimensions)}});
    } break;
    case FLOAT64: {
      double *result = safe_mal<double>(size);
      chooseExecutionMethod(curr, predData, result, size);
      results.insert(
          {curr,
           {.data = (void *)result,
            .type = FLOAT64,
            .num_entries = size,
            .shape = vector<size_t>(curr->operation->shape,
                                    curr->operation->shape +
                                        curr->operation->dimensions)}});
    } break;
    }
  }
  CPUResultData final = results[node];
  // free all other data
  for (auto &[gn, rd] : results) {
    if (gn != node)
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
  op->op_type = RESULTDATA;
  FGraphNode *rn = new FGraphNode();
  rn->operation = op;
  rn->predecessors = safe_mal<FGraphNode *>(1);
  rn->predecessors[0] = node;
  node->reference_counter++;
  rn->num_predecessor = 1;
  rn->reference_counter = 0;
  return rn;
}
