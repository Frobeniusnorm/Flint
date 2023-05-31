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

  This file includes the implementation of the thunk generation of the CPU backend.
*/
#include "src/cputhunk.hpp"
#include "flint.h"
#include "src/utils.hpp"
#include <cstdlib>
#include <cstring>
#include <list>
#include <unordered_map>
#include <unordered_set>
template<typename R>
CPUThunk genFunction(FGraphNode* node, std::vector<CPUThunk> pred) {
  switch (node->operation->op_type) {
  case FSTORE: {
    return CPUThunk{.node = node, .calculate = [](CPUThunk* self, void* result, size_t index){
      const FStore* store = (FStore*) self->node->operation->additional_data;
          *((R*) result) = ((R*) store->data)[index];
          return;
    }};
  }
  case FGEN_RANDOM:
    return CPUThunk{.node = node, .calculate = [](CPUThunk* self, void* result, size_t index){
      *((double*) result) = (double)rand() / (double)RAND_MAX;
    }};
    default: return {};
  }
}
template<typename R, typename A>
CPUThunk unaryFunction(FGraphNode* node, std::vector<CPUThunk> pred) {
  switch (node->operation->op_type) {
  case FLATTEN:
  case FNEG:
  case FLOG:
  case FSIGN:
  case FEVEN:
  case FLOG2:
  case FLOG10:
  case FSIN:
  case FCOS:
  case FTAN:
  case FASIN:
  case FACOS:
  case FATAN:
  case FSQRT:
  case FCONVERSION:
  case FRESHAPE:
  case FREDUCE_SUM:
  case FREDUCE_MUL:
  case FSLICE:
  case FABS:
  case FREPEAT:
  case FTRANSPOSE:
  case FEXTEND:
    default: return {};
  }
}
template<typename R, typename A, typename B>
CPUThunk binaryFunction(FGraphNode* node, std::vector<CPUThunk> pred) {
  switch (node->operation->op_type) {
  case FADD:
  case FSUB:
  case FMUL:
  case FDIV:
  case FPOW:
  case FMATMUL:
  case FMIN:
  case FMAX:
  case FLESS:
  case FEQUAL:
  case FGREATER:
  case FCONVOLVE:
  case FSLIDE:
  case FGRADIENT_CONVOLVE:
    default: return {};
  }
}
template <typename R, typename A>
inline CPUThunk unaryTemp0(FGraphNode *node, std::vector<CPUThunk> pred){
  switch (node->predecessors[1]->operation->data_type){
  case F_INT32:
    return unaryFunction<R, int>(node, pred);
  case F_INT64:
    return unaryFunction<R, long>(node, pred);
  case F_FLOAT32:
    return unaryFunction<R, float>(node, pred);
  case F_FLOAT64:
    return unaryFunction<R, double>(node, pred);
  }
}
template <typename R, typename A>
inline CPUThunk binaryTemp0(FGraphNode *node, std::vector<CPUThunk> pred){
  switch (node->predecessors[1]->operation->data_type){
  case F_INT32:
    return binaryFunction<R, A, int>(node, pred);
  case F_INT64:
    return binaryFunction<R, A, long>(node, pred);
  case F_FLOAT32:
    return binaryFunction<R, A, float>(node, pred);
  case F_FLOAT64:
    return binaryFunction<R, A, double>(node, pred);
  }
}
template <typename R>
inline CPUThunk binaryTemp1(FGraphNode *node, std::vector<CPUThunk> pred){
  switch (node->predecessors[0]->operation->data_type){
  case F_INT32:
    return binaryTemp0<R, int>(node, pred);
  case F_INT64:
    return binaryTemp0<R, long>(node, pred);
  case F_FLOAT32:
    return binaryTemp0<R, float>(node, pred);
  case F_FLOAT64:
    return binaryTemp0<R, double>(node, pred);
  }
}
CPUThunk buildThunk(FGraphNode *node, std::vector<CPUThunk> pred) {
  if (pred.size() == 2) {
    switch (node->operation->data_type) {
    case F_INT32:
      return binaryTemp1<int>(node, pred);
    case F_INT64:
      return binaryTemp1<long>(node, pred);
    case F_FLOAT32:
      return binaryTemp1<float>(node, pred);
    case F_FLOAT64:
      return binaryTemp1<double>(node, pred);
    }
  } else if (pred.size() == 1) {
    switch (node->operation->data_type) {
    case F_INT32:
      return unaryTemp0<int>(node, pred);
    case F_INT64:
      return unaryTemp0<long>(node, pred);
    case F_FLOAT32:
      return unaryTemp0<float>(node, pred);
    case F_FLOAT64:
      return unaryTemp0<double>(node, pred);
    }
  } else {
    switch (node->operation->data_type) {
    case F_INT32:
      return genFunction<int>(node, pred);
    case F_INT64:
      return genFunction<long>(node, pred);
    case F_FLOAT32:
      return genFunction<float>(node, pred);
    case F_FLOAT64:
      return genFunction<double>(node, pred);
    }
  }
}
CPUThunk generateThunk(FGraphNode *node) {
  using namespace std;
  unordered_set<FGraphNode *> inExecuteList;
  list<FGraphNode *> workList;  // traverse bottom up
  list<FGraphNode *> toExecute; // in top down order
  workList.push_front(node);
  // collect nodes
  while (!workList.empty()) {
    FGraphNode *curr = workList.front();
    workList.pop_front();
    if (inExecuteList.find(curr) != inExecuteList.end())
      toExecute.remove(curr);
    inExecuteList.insert(curr);
    toExecute.push_front(curr);
    for (int i = 0; i < curr->num_predecessor; i++) {
      // execute on GPU if it makes more sense
      if (flintInitializedBackends() & FLINT_BACKEND_ONLY_GPU) {
        FGraphNode *p = curr->predecessors[i];
        size_t score = computeScore(p, true);
        if (score >= 2048) {
          if (inExecuteList.find(p) != inExecuteList.end())
            toExecute.remove(p);
          fSyncMemory(fExecuteGraph_gpu(p));
          toExecute.push_front(p);
          inExecuteList.insert(p);
          continue;
        }
      }
      workList.push_back(curr->predecessors[i]);
    }
  }
  unordered_map<FGraphNode *, CPUThunk> results;
  // work them in correct oder
  for (FGraphNode *curr : toExecute) {
    vector<CPUThunk> predData(curr->num_predecessor);
    for (int i = 0; i < curr->num_predecessor; i++) {
      predData[i] = results[curr->predecessors[i]];
    }
    results.insert({curr, buildThunk(node, predData)});
  }
  return results[node];
}
