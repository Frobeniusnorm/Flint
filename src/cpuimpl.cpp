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
#include <cmath>
#include <cstring>
#include <list>
#include <queue>
#include <unordered_map>
static bool initialized = false;

void flintInit_cpu() {
  if (!initialized) {
    initialized = true;
  }
  // TODO
}

struct CPUResultData {
  void *data;
  FType type;
  int num_entries;
  std::vector<int> shape;
};
template <typename T, typename A, typename B>
static void binaryExpression(T *result, A *data1, B *data2, FOperationType op,
                             int from, int size, int index_man_1,
                             int index_man_2) {
  switch (op) {
  case ADD:
    for (int i = from; i < from + size; i++)
      result[i] = data1[i % index_man_1] + data2[i % index_man_2];
    break;
  case SUB:
    for (int i = from; i < from + size; i++)
      result[i] = data1[i % index_man_1] - data2[i % index_man_2];
    break;
  case MUL:
    for (int i = from; i < from + size; i++)
      result[i] = data1[i % index_man_1] * data2[i % index_man_2];
    break;
  case DIV:
    for (int i = from; i < from + size; i++)
      result[i] = data1[i % index_man_1] / data2[i % index_man_2];
    break;
  case POW:
    for (int i = from; i < from + size; i++)
      result[i] = pow(data1[i % index_man_1], data2[i % index_man_2]);
    break;
  default:
    break;
  }
}
// i hate this function more than anything else in this library (yet)
template <typename T>
static void executeNode(FGraphNode *node,
                        std::vector<CPUResultData> predecessor_data, T *result,
                        int from, int size) {
  switch (node->operation->op_type) {
  case STORE: {
    FStore *store = (FStore *)node->operation->additional_data;
    for (int i = from; i < from + size; i++)
      result[i] = ((T *)store->data)[i];
  } break;
  case RESULTDATA: {
    FResultData *store = (FResultData *)node->operation->additional_data;
    for (int i = from; i < from + size; i++)
      result[i] = ((T *)store->data)[i];
  } break;
  case CONST: {
    FConst *cons = (FConst *)node->operation->additional_data;
    result[from] = ((T *)cons->value)[0];
  } break;
  default: { // binary operations
    CPUResultData p1 = predecessor_data[0], p2 = predecessor_data[1];
    int im1 = p1.num_entries, im2 = p2.num_entries;

    switch (p1.type) {
    case INT32:
      switch (p2.type) {
      case INT32:
        binaryExpression(result, (int *)p1.data, (int *)p2.data,
                         node->operation->op_type, from, size, im1, im2);
        break;
      case FLOAT32:
        binaryExpression(result, (int *)p1.data, (float *)p2.data,
                         node->operation->op_type, from, size, im1, im2);
        break;
      case FLOAT64:
        binaryExpression(result, (int *)p1.data, (double *)p2.data,
                         node->operation->op_type, from, size, im1, im2);
        break;
      case INT64:
        binaryExpression(result, (int *)p1.data, (long *)p2.data,
                         node->operation->op_type, from, size, im1, im2);
        break;
      }
      break;
    case FLOAT32:
      switch (p2.type) {
      case INT32:
        binaryExpression(result, (float *)p1.data, (int *)p2.data,
                         node->operation->op_type, from, size, im1, im2);
        break;
      case FLOAT32:
        binaryExpression(result, (float *)p1.data, (float *)p2.data,
                         node->operation->op_type, from, size, im1, im2);
        break;
      case FLOAT64:
        binaryExpression(result, (float *)p1.data, (double *)p2.data,
                         node->operation->op_type, from, size, im1, im2);
        break;
      case INT64:
        binaryExpression(result, (float *)p1.data, (long *)p2.data,
                         node->operation->op_type, from, size, im1, im2);
        break;
      }
      break;
    case FLOAT64:
      switch (p2.type) {
      case INT32:
        binaryExpression(result, (double *)p1.data, (int *)p2.data,
                         node->operation->op_type, from, size, im1, im2);
        break;
      case FLOAT32:
        binaryExpression(result, (double *)p1.data, (float *)p2.data,
                         node->operation->op_type, from, size, im1, im2);
        break;
      case FLOAT64:
        binaryExpression(result, (double *)p1.data, (double *)p2.data,
                         node->operation->op_type, from, size, im1, im2);
        break;
      case INT64:
        binaryExpression(result, (double *)p1.data, (long *)p2.data,
                         node->operation->op_type, from, size, im1, im2);
        break;
      }
      break;
    case INT64:
      switch (p2.type) {
      case INT32:
        binaryExpression(result, (long *)p1.data, (int *)p2.data,
                         node->operation->op_type, from, size, im1, im2);
        break;
      case FLOAT32:
        binaryExpression(result, (long *)p1.data, (float *)p2.data,
                         node->operation->op_type, from, size, im1, im2);
        break;
      case FLOAT64:
        binaryExpression(result, (long *)p1.data, (double *)p2.data,
                         node->operation->op_type, from, size, im1, im2);
        break;
      case INT64:
        binaryExpression(result, (long *)p1.data, (long *)p2.data,
                         node->operation->op_type, from, size, im1, im2);
        break;
      }
      break;
    }
    break;
  }
  }
}

FGraphNode *executeGraph_cpu(FGraphNode *node) {
  if (!initialized)
    flintInit_cpu();
  // TODO parallel execution
  using namespace std;
  unordered_map<FGraphNode *, CPUResultData> results;
  queue<FGraphNode *> workList; // traverse bottom up
  list<FGraphNode *> toExecute; // in top down order
  // collect nodes
  while (!workList.empty()) {
    FGraphNode *curr = workList.front();
    workList.pop();
    toExecute.push_front(curr);
    for (int i = 0; i < curr->num_predecessor; i++)
      workList.push(curr->predecessors[i]);
  }
  // work them in correct oder
  for (FGraphNode *curr : toExecute) {
    // collect predecessor results
    vector<CPUResultData> predData(curr->num_predecessor);
    for (int i = 0; i < curr->num_predecessor; i++) {
      if (results.find(curr->predecessors[i]) == results.end())
        log(ERROR, "oh i fucked up");
      predData[i] = results[curr->predecessors[i]];
    }
    // calculate total size
    int size = 1;
    for (int j = 0; j < curr->operation->dimensions; j++)
      size *= curr->operation->shape[j];
    // allocate result data and execute
    switch (curr->operation->data_type) {
    case INT32: {
      int *result = (int *)malloc(size * sizeof(int));
      executeNode(curr, predData, result, 0, size);
      results.insert({curr,
                      {.data = (void *)result,
                       .type = INT32,
                       .num_entries = size,
                       .shape = vector<int>(curr->operation->shape,
                                            curr->operation->shape +
                                                curr->operation->dimensions)}});
    } break;
    case INT64: {
      long *result = (long *)malloc(size * sizeof(long));
      executeNode(curr, predData, result, 0, size);
      results.insert({curr,
                      {.data = (void *)result,
                       .type = INT64,
                       .num_entries = size,
                       .shape = vector<int>(curr->operation->shape,
                                            curr->operation->shape +
                                                curr->operation->dimensions)}});
    } break;
    case FLOAT32: {
      float *result = (float *)malloc(size * sizeof(float));
      executeNode(curr, predData, result, 0, size);
      results.insert({curr,
                      {.data = (void *)result,
                       .type = FLOAT32,
                       .num_entries = size,
                       .shape = vector<int>(curr->operation->shape,
                                            curr->operation->shape +
                                                curr->operation->dimensions)}});
    } break;
    case FLOAT64: {
      double *result = (double *)malloc(size * sizeof(double));
      executeNode(curr, predData, result, 0, size);
      results.insert({curr,
                      {.data = (void *)result,
                       .type = FLOAT64,
                       .num_entries = size,
                       .shape = vector<int>(curr->operation->shape,
                                            curr->operation->shape +
                                                curr->operation->dimensions)}});
    } break;
    }
  }
  CPUResultData final = results[node];
  FResultData *rd = new FResultData();
  rd->data = final.data;
  rd->num_entries = final.num_entries;
  rd->mem_id = nullptr;
  FOperation *op = new FOperation();
  op->dimensions = node->operation->dimensions;
  op->shape = safe_mal<int>(final.shape.size() * sizeof(int));
  memcpy(op->shape, final.shape.data(), final.shape.size() * sizeof(int));
  op->data_type = node->operation->data_type;
  op->additional_data = (void *)rd;
  op->op_type = RESULTDATA;
  FGraphNode *rn = new FGraphNode();
  rn->operation = op;
  rn->predecessors = safe_mal<FGraphNode *>(1);
  rn->predecessors[0] = node;
  rn->num_predecessor = 1;
  rn->reference_counter = 1;
  return rn;
}
