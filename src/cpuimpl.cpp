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
#include <iostream>
#include <list>
#include <queue>
#include <semaphore>
#include <stdlib.h>
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
    flogging(F_INFO,
             "Using " + std::to_string(cores) + " threads for CPU-backend");
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
static void binaryExpression(T *__restrict__ result, const A *__restrict__ data1,
                             const B *__restrict__ data2, FOperationType op, size_t from,
                             size_t size, int index_man_1, int index_man_2,
                             const FGraphNode *curr) {
  switch (op) {
  case FADD:
    for (size_t i = from; i < from + size; i++) {
      result[i] = data1[i % index_man_1] + data2[i % index_man_2];
    }
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
        result[index] += data1[base_p1 + j * m + i] * data2[base_p2 + i * n + k];
      }
    }
  } break;
  case FGRADIENT_CONVOLVE: {
    const FOperation *op = curr->operation;
    const FGraphNode *gnp1 = curr->predecessors[0],
                     *gnp2 = curr->predecessors[1];
    const FOperation *kernel = gnp1->operation, *a = gnp2->operation;
    const unsigned int *steps = (unsigned int *)op->additional_data;
    // calculate accumulated sizes for result (pred), kernel and a (adjacent)
    std::vector<size_t> acc_sizes(op->dimensions - 1);
    std::vector<size_t> acc_sizes_pred(op->dimensions);
    std::vector<size_t> acc_sizes_kernel(op->dimensions);
    acc_sizes_kernel[op->dimensions - 1] = 1;
    acc_sizes_pred[op->dimensions - 1] = 1;
    acc_sizes[op->dimensions - 2] = 1;
    size_t kernel_num_elems = kernel->shape[op->dimensions - 1];
    for (long d = op->dimensions - 2; d >= 0; d--) {
      kernel_num_elems *= kernel->shape[d];
      acc_sizes_kernel[d] = acc_sizes_kernel[d + 1] * kernel->shape[d + 1];
      acc_sizes_pred[d] = acc_sizes_pred[d + 1] * op->shape[d + 1];
    }
    for (long d = op->dimensions - 3; d >= 0; d--)
      acc_sizes[d] = acc_sizes[d + 1] * a->shape[d + 1];

    for (size_t i = from; i < from + size; i++) {
      T res = 0;
      long k = 0;
      bool in_steps = true;
      // reproject first time kernel hits i
      for (int d = op->dimensions - 1; d >= 0; d--) {
        size_t di =
            (d == 0 ? i : i % acc_sizes_pred[d - 1]) / acc_sizes_pred[d];
        size_t dk = d == op->dimensions - 1 ? di : di % steps[d];
        if (dk >= kernel->shape[d]) {
          in_steps = false;
          break;
        }
        k += dk * acc_sizes_kernel[d];
      }
      if (in_steps)
        while (k < kernel_num_elems) {
          size_t i_conv = 0;
          for (int d = 0; d < op->dimensions - 2; d++) {
            const size_t dk = (d == 0 ? k : k % acc_sizes_kernel[d - 1]) /
                              acc_sizes_kernel[d];
            const size_t di =
                (d == 0 ? i : i % acc_sizes_pred[d - 1]) / acc_sizes_pred[d];
            const size_t j = (di - dk) / steps[d]; // top left corner
            i_conv += j * acc_sizes[d];
          }
          res += data1[k] * data2[i_conv];
          long step = 0;
          // reproject index to calculate step from steps
          for (int d = op->dimensions - 2; d >= 0; d--) {
            const int stepd = steps[d];
            const size_t dk = (d == 0 ? k : k % acc_sizes_kernel[d - 1]) /
                              acc_sizes_kernel[d];
            const size_t di =
                (d == 0 ? i : i % acc_sizes_pred[d - 1]) / acc_sizes_pred[d];
            if (dk + stepd < kernel->shape[d] && di >= dk + stepd) {
              step += stepd * acc_sizes_kernel[d];
              break;
            } else {
              step -=
                  (dk - (di % stepd)) *
                  acc_sizes_kernel[d]; // set to kernel start in this dimension
            }
          }
          if (step <= 0)
            break; // total overflow
          k += step;
        }
      result[i] = res;
    }
  } break;
  case FCONVOLVE: {
    const FOperation *op = curr->operation;
    const FGraphNode *gnp1 = curr->predecessors[0],
                     *gnp2 = curr->predecessors[1];
    const FOperation *pred = gnp1->operation, *kernel = gnp2->operation;
    const unsigned int *steps = (unsigned int *)op->additional_data;
    // calculate accumulated sizes for result, kernel and source (pred)
    std::vector<size_t> acc_sizes(op->dimensions);
    std::vector<size_t> acc_sizes_pred(acc_sizes.size() + 1);
    std::vector<size_t> acc_sizes_kernel(acc_sizes.size() + 1);
    acc_sizes[op->dimensions - 1] = 1;
    for (long d = op->dimensions - 2; d >= 0; d--) {
      acc_sizes[d] = acc_sizes[d + 1] * op->shape[d + 1];
    }
    acc_sizes_kernel[acc_sizes.size()] = 1;
    acc_sizes_pred[acc_sizes.size()] = 1;
    size_t kernel_num_elems = kernel->shape[acc_sizes.size()];
    size_t pred_num_elems = pred->shape[acc_sizes.size()];
    for (long d = acc_sizes.size() - 1; d >= 0; d--) {
      pred_num_elems *= pred->shape[d];
      kernel_num_elems *= kernel->shape[d];
      acc_sizes_kernel[d] = acc_sizes_kernel[d + 1] * kernel->shape[d + 1];
      acc_sizes_pred[d] = acc_sizes_pred[d + 1] * pred->shape[d + 1];
    }
    for (size_t i = from; i < from + size; i++) {
      size_t j = 0;
      // we can ignore last index of source and kernel for result since we
      // iterate over it (i.e. for the destination it is 0 since it does not
      // have that dimension)
      for (unsigned int d = 0; d < op->dimensions; d++) {
        // get dimension index
        size_t di = (d == 0 ? i : i % acc_sizes[d - 1]) / acc_sizes[d];
        // reproject
        j += di * steps[d] * acc_sizes_pred[d];
      }
      // now that we have the correct base index in source, convolve
      T res = 0;
      for (size_t k = 0; k < kernel_num_elems; k++) {
        bool set_zero = false;
        size_t o = 0; // source offset
        // reproject kernel
        for (unsigned int d = 0; d < acc_sizes_kernel.size(); d++) {
          size_t di = d == acc_sizes_kernel.size() - 1
                          ? 0
                          : (d == 0 ? i : i % acc_sizes[d - 1]) / acc_sizes[d];
          size_t dk =
              (d == 0 ? k : k % acc_sizes_kernel[d - 1]) / acc_sizes_kernel[d];
          if (d < op->dimensions)
            if (((di * steps[d]) + dk) * acc_sizes_pred[d] >= pred_num_elems ||
                (d > 0 && ((di * steps[d]) + dk) * acc_sizes_pred[d] >=
                              acc_sizes_pred[d - 1])) {
              set_zero = true;
              break;
            }
          o += dk * acc_sizes_pred[d];
        }
        if (set_zero)
          continue;
        res += data2[k] * data1[j + o];
      }
      result[i] = res;
    }

  } break;
  case FSLIDE: {
    const FOperation *op = curr->operation;
    const FGraphNode *gnp1 = curr->predecessors[0],
                     *gnp2 = curr->predecessors[1];
    const FOperation *pred = gnp1->operation, *kernel = gnp2->operation;
    std::vector<size_t> acc_sizes_pred(pred->dimensions);
    std::vector<size_t> acc_sizes_kernel(kernel->dimensions);
    acc_sizes_pred[pred->dimensions - 1] = 1;
    acc_sizes_kernel[kernel->dimensions - 1] = 1;
    size_t pred_num_elems = pred->shape[pred->dimensions - 1];
    for (long d = pred->dimensions - 2; d >= 0; d--) {
      pred_num_elems *= pred->shape[d];
      acc_sizes_pred[d] = acc_sizes_pred[d + 1] * pred->shape[d + 1];
      acc_sizes_kernel[d] = acc_sizes_kernel[d + 1] * kernel->shape[d + 1];
    }
    const unsigned int *steps = (unsigned int *)op->additional_data;
    for (size_t i = from; i < from + size; i++) {
      size_t a = 0;
      // reproject start
      for (int d = kernel->dimensions - 1; d >= 0; d--) {
        size_t di =
            (d == 0 ? i : i % acc_sizes_kernel[d - 1]) / acc_sizes_kernel[d];
        a += di * acc_sizes_pred[d];
      }
      T res = 0;
      // we want to iterate over all elements it would be slid agains
      while (a < pred_num_elems) {
        long step = 0;
        res += data1[a] * data2[i];
        // reproject index to calculate step from steps
        for (int d = pred->dimensions - 2; d >= 0; d--) {
          size_t da =
              (d == 0 ? a : a % acc_sizes_pred[d - 1]) / acc_sizes_pred[d];
          if (da + steps[d] < pred->shape[d]) {
            step += steps[d] * acc_sizes_pred[d];
            break;
          } else {
            size_t di = (d == 0 ? i : i % acc_sizes_kernel[d - 1]) /
                        acc_sizes_kernel[d];
            step -= (da - di) *
                    acc_sizes_pred[d]; // set to kernel start in this dimension
          }
        }
        if (step <= 0)
          break; // total overflow
        a += step;
      }
      result[i] = res;
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
  case FEQUAL:
    for (size_t i = from; i < from + size; i++)
      result[i] = data1[i % index_man_1] == data2[i % index_man_2] ? 1 : 0;
    break;
  case FLESS:
    for (size_t i = from; i < from + size; i++)
      result[i] = data1[i % index_man_1] < data2[i % index_man_2] ? 1 : 0;
    break;
  case FGREATER:
    for (size_t i = from; i < from + size; i++)
      result[i] = data1[i % index_man_1] > data2[i % index_man_2] ? 1 : 0;
    break;
  default:
    break;
  }
}
// i hate this function more than anything else in this library (yet)
// EDIT: nope i was wrong, computeGradient is worse
template <typename T>
static void executeNode(const FGraphNode *node,
                        std::vector<CPUResultData> predecessor_data,
                        T *__restrict__ result, size_t from, size_t size) {
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
  case FREPEAT: {
    const FOperation *op = node->operation;
    const CPUResultData pred = predecessor_data[0];
    const void *__restrict__ data = pred.data;
    for (int i = from; i < from + size; i++) {
      // calculate number of elements per dimension entry for destination and
      // source
      std::vector<size_t> acc_sizes_d(op->dimensions);
      std::vector<size_t> acc_sizes_s(op->dimensions);
      acc_sizes_d[op->dimensions - 1] = 1;
      acc_sizes_s[op->dimensions - 1] = 1;
      for (int dim = op->dimensions - 2; dim >= 0; dim--) {
        acc_sizes_d[dim] = acc_sizes_d[dim + 1] * op->shape[dim + 1];
        acc_sizes_s[dim] = acc_sizes_s[dim + 1] * pred.shape[dim + 1];
      }
      // to get the index in the source array we first calculate the indices and
      // reproject
      int index = i;
      int src_index = 0;
      for (int dim = 0; dim < op->dimensions; dim++) {
        int curr_idx = index / acc_sizes_d[dim];
        index %= acc_sizes_d[dim];
        src_index += (curr_idx % pred.shape[dim]) * acc_sizes_s[dim];
      }
      result[i] = ((const T *__restrict__)data)[src_index];
    }
  } break;
  case FTRANSPOSE: {
    const FOperation *op = node->operation;
    const int *transposition = (int *)op->additional_data;
    CPUResultData pred = predecessor_data[0];
    const void *__restrict__ data = pred.data;
    // calculate number of elements per dimension entry for destination and
    // source
    std::vector<size_t> acc_sizes_d(op->dimensions);
    std::vector<size_t> acc_sizes_s(op->dimensions);
    acc_sizes_d[op->dimensions - 1] = 1;
    acc_sizes_s[op->dimensions - 1] = 1;
    for (int dim = op->dimensions - 2; dim >= 0; dim--) {
      acc_sizes_d[dim] = acc_sizes_d[dim + 1] * op->shape[dim + 1];
      acc_sizes_s[dim] = acc_sizes_s[dim + 1] * pred.shape[dim + 1];
    }
    for (int i = from; i < from + size; i++) {
      // to get the index in the source array we first calculate the indices and
      // reproject
      int index = i;
      int src_index = 0;
      for (int dim = 0; dim < op->dimensions; dim++) {
        int curr_idx = index / acc_sizes_d[dim];
        index %= acc_sizes_d[dim];
        src_index += curr_idx * acc_sizes_s[transposition[dim]];
      }
      result[i] = ((const T *__restrict__)data)[src_index];
    }
  } break;
  case FREDUCE_SUM:
  case FREDUCE_MUL: {
    const CPUResultData pred = predecessor_data[0];
    const int dim = ((int *)node->operation->additional_data)[0];
    const void *__restrict__ data = pred.data;
    size_t it_dim = 1; // iteration size <=> product of all dimensions along dim
    for (size_t d = dim + 1; d < pred.shape.size(); d++)
      it_dim *= pred.shape[d];

    for (size_t i = from; i < from + size; i++) {
      // iterate through to-reduce dimension
      result[i] = node->operation->op_type == FREDUCE_SUM
                      ? 0
                      : 1; // init with neutral element
      for (size_t j = 0; j < pred.shape[dim]; j++) {
        const T curr = ((const T *__restrict__)
                            data)[(i / it_dim) * it_dim * pred.shape[dim] +
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
    const void *__restrict__ data = pred.data;
    for (size_t i = from; i < from + size; i++)
      result[i] = ((const T *__restrict__)data)[i];
  } break;
  case FSLICE: {
    CPUResultData pred = predecessor_data[0];
    FSlice *slice = (FSlice *)node->operation->additional_data;
    const void *__restrict__ data = pred.data;
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
    for (unsigned int d = 0; d < node->operation->dimensions; d++) {
      start += slice->start[d] * acc_sizes_pred[d];
    }
    // calculate for each entry corresponding element
    for (size_t i = from; i < from + size; i++) {
      size_t j = start;
      for (unsigned int d = 0; d < node->operation->dimensions; d++) {
        // get dimension index
        size_t di = (d == 0 ? i : i % acc_sizes[d - 1]) / acc_sizes[d];
        // reproject
        j += di * slice->step[d] * acc_sizes_pred[d];
      }
      result[i] = ((const T *__restrict__)data)[j];
    }
  } break;
  case FEXTEND: {
    CPUResultData pred = predecessor_data[0];
    const void *__restrict__ data = pred.data;
    FExtend *extend = (FExtend *)node->operation->additional_data;
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
    // calculate for each entry corresponding element
    for (size_t i = from; i < from + size; i++) {
      size_t j = 0;
      bool set_zero = false;
      for (size_t d = 0; d < acc_sizes.size(); d++) {
        long step = extend->step[d];
        bool inv = step < 0;
        if (inv)
          step = -step;
        // get dimension index
        size_t di = (d == 0 ? i : i % acc_sizes[d - 1]) / acc_sizes[d];
        if (di < extend->start[d]) {
          set_zero = true;
          break;
        }
        di -= extend->start[d];
        if (di % step != 0) {
          set_zero = true;
          break;
        }
        di /= step;
        if (di >= pred.shape[d]) {
          set_zero = true;
          break;
        }
        // reverse if negative
        if (inv) {
          di = pred.shape[d] - di - 1;
        }
        // reproject
        j += di * acc_sizes_pred[d];
      }
      result[i] = set_zero ? 0 : ((const T *__restrict__)data)[j];
    }
  } break;
  case FABS: {
    CPUResultData pred = predecessor_data[0];
    const void *__restrict__ data = pred.data;
    for (size_t i = from; i < from + size; i++)
      result[i] = abs(((const T *__restrict__)data)[i]);
  } break;
  case FLOG: {
    CPUResultData pred = predecessor_data[0];
    const void *__restrict__ data = pred.data;
    for (size_t i = from; i < from + size; i++) {
      result[i] = log(((const T *__restrict__)data)[i]);
    }
  } break;
  case FLOG2: {
    CPUResultData pred = predecessor_data[0];
    const void *__restrict__ data = pred.data;
    for (size_t i = from; i < from + size; i++)
      result[i] = log2(((const T *__restrict__)data)[i]);
  } break;
  case FNEG: {
    CPUResultData pred = predecessor_data[0];
    const void *__restrict__ data = pred.data;
    for (size_t i = from; i < from + size; i++)
      result[i] = -((const T *__restrict__)data)[i];
  } break;
  case FSIGN: {
    const CPUResultData pred = predecessor_data[0];
    switch (pred.type) {
    case F_INT32: {
      for (size_t i = from; i < from + size; i++) {
        int val = ((const int *__restrict__)pred.data)[i];
        result[i] = val < 0 ? -1 : 1;
      }
    } break;
    case F_INT64: {
      for (size_t i = from; i < from + size; i++) {
        long val = ((const long *__restrict__)pred.data)[i];
        result[i] = val < 0 ? -1 : 1;
      }
    } break;
    case F_FLOAT32: {
      for (size_t i = from; i < from + size; i++) {
        float val = ((const float *__restrict__)pred.data)[i];
        result[i] = val < 0 ? -1 : 1;
      }
    } break;
    case F_FLOAT64: {
      for (size_t i = from; i < from + size; i++) {
        double val = ((const double *__restrict__)pred.data)[i];
        result[i] = val < 0 ? -1 : 1;
      }
    } break;
    }
  } break;
  case FEVEN: { // it aint pretty but it does its job
    const CPUResultData pred = predecessor_data[0];
    switch (pred.type) {
    case F_INT32: {
      for (size_t i = from; i < from + size; i++) {
        int val = ((const int *__restrict__)pred.data)[i];
        result[i] = val % 2 == 0 ? 1 : 0;
      }
    } break;
    case F_INT64: {
      for (size_t i = from; i < from + size; i++) {
        long val = ((const long *__restrict__)pred.data)[i];
        result[i] = val % 2 == 0 ? 1 : 0;
      }
    } break;
    case F_FLOAT64:
    case F_FLOAT32: {
      for (size_t i = from; i < from + size; i++) {
        result[i] = 0;
      }
    } break;
    }
  } break;
  case FLOG10: {
    const CPUResultData pred = predecessor_data[0];
    for (size_t i = from; i < from + size; i++)
      result[i] = log10(((const T *__restrict__)pred.data)[i]);
  } break;
  case FSIN: {
    const CPUResultData pred = predecessor_data[0];
    for (size_t i = from; i < from + size; i++)
      result[i] = sin(((const T *__restrict__)pred.data)[i]);
  } break;
  case FSQRT: {
    const CPUResultData pred = predecessor_data[0];
    for (size_t i = from; i < from + size; i++)
      result[i] = sqrt(((const T *__restrict__)pred.data)[i]);
  } break;
  case FCOS: {
    const CPUResultData pred = predecessor_data[0];
    for (size_t i = from; i < from + size; i++)
      result[i] = cos(((const T *__restrict__)pred.data)[i]);
  } break;
  case FTAN: {
    const CPUResultData pred = predecessor_data[0];
    for (size_t i = from; i < from + size; i++)
      result[i] = tan(((const T *__restrict__)pred.data)[i]);
  } break;
  case FASIN: {
    const CPUResultData pred = predecessor_data[0];
    for (size_t i = from; i < from + size; i++)
      result[i] = asin(((const T *__restrict__)pred.data)[i]);
  } break;
  case FACOS: {
    const CPUResultData pred = predecessor_data[0];
    for (size_t i = from; i < from + size; i++)
      result[i] = acos(((const T *__restrict__)pred.data)[i]);
  } break;
  case FATAN: {
    const CPUResultData pred = predecessor_data[0];
    for (size_t i = from; i < from + size; i++)
      result[i] = atan(((const T *__restrict__)pred.data)[i]);
  } break;
  default: { // binary operations
    const CPUResultData p1 = predecessor_data[0], p2 = predecessor_data[1];
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
    flogging(F_DEBUG, "Sending kill signal and poisson pills");
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
#define PARALLEL_EXECUTION_SIZE 64 // for debugging
template <typename T>
inline void chooseExecutionMethod(FGraphNode *node,
                                  std::vector<CPUResultData> pred_data,
                                  T *result, size_t size) {
  auto start = std::chrono::high_resolution_clock::now();
  size_t score = size * operationScore(node);
  if (score >= PARALLEL_EXECUTION_SIZE) {
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
  flogging(F_DEBUG, (size >= PARALLEL_EXECUTION_SIZE
                         ? std::string("Parallel Execution on CPU (score: " +
                                       std::to_string(score) + ")")
                         : std::string("Sequential Execution on CPU (score: " +
                                       std::to_string(score) + ")")) +
                        " took " + std::to_string(elapsed.count()) + "ms");
}
FGraphNode *fExecuteGraph_cpu_eagerly(FGraphNode *node) {
  if (!initialized)
    flintInit_cpu();
  if (node->result_data)
    return node;
  bool is_data_node = node->operation->op_type == FSTORE;
  std::vector<CPUResultData> pred_data(node->num_predecessor);
  size_t total = 1;
  for (int i = 0; i < node->operation->dimensions; i++)
    total *= node->operation->shape[i];
  void *data;

  if (!is_data_node) {
    // build predecessor data
    for (int i = 0; i < node->num_predecessor; i++) {
      FGraphNode *pred = node->predecessors[i];
      if (pred->operation->op_type == FSTORE) {
        FStore *store = (FStore *)pred->operation->additional_data;
        pred_data[i].data = store->data;
        pred_data[i].num_entries = store->num_entries;
      } else if (pred->result_data) {
        pred_data[i].data = pred->result_data->data;
        pred_data[i].num_entries = pred->result_data->num_entries;
      } else { // FConst
        pred_data[i].num_entries = 1;
        pred_data[i].data = ((FConst *)pred->operation)->value;
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
    data = ((FStore *)node->operation->additional_data)->data;
  }
  FResultData *rd = new FResultData();
  rd->data = data;
  rd->num_entries = total;
  rd->mem_id = nullptr;
  node->result_data = rd;
  return node;
}

FGraphNode *fExecuteGraph_cpu(FGraphNode *node) {
  if (!initialized)
    flintInit_cpu();
  if (node->result_data)
    return node;
  if (node->operation->op_type == FSTORE) {
    node->result_data = new FResultData();
    FStore *store = (FStore *)node->operation->additional_data;
    node->result_data->num_entries = store->num_entries;
    node->result_data->mem_id = store->mem_id;
    node->result_data->data = store->data;
    return node;
  }
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
    if (curr->operation->op_type == FSTORE || curr->result_data) {
      CPUResultData foo;
      foo.shape =
          vector<size_t>(curr->operation->shape,
                         curr->operation->shape + curr->operation->dimensions);
      foo.type = curr->operation->data_type;
      if (curr->result_data) {
        FResultData *rd = curr->result_data;
        foo.num_entries = rd->num_entries;
        foo.data = rd->data;
      } else {
        FStore *store = (FStore *)curr->operation->additional_data;
        foo.num_entries = store->num_entries;
        foo.data = store->data;
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
  if (!fIsEagerExecution()) {
    // free all other data
    for (auto &[gn, rd] : results) {
      if (gn != node && gn->operation->op_type != FSTORE && !gn->result_data)
        free(rd.data);
    }
  } else {
    for (auto &[gn, rd] : results) {
      if (gn != node && gn->operation->op_type != FSTORE && !gn->result_data) {
        FResultData *result = new FResultData();
        result->data = rd.data;
        result->num_entries = rd.num_entries;
        result->mem_id = nullptr;
        gn->result_data = result;
      }
    }
  }
  // return data
  FResultData *rd = new FResultData();
  rd->data = final.data;
  rd->num_entries = final.num_entries;
  rd->mem_id = nullptr;
  node->result_data = rd;
  return node;
}
