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

#ifndef UTILS_HPP
#define UTILS_HPP
#include "../flint.h"
#include <cmath>
#include <condition_variable>
#include <limits>
#include <list>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <vector>

template <typename T> inline T *safe_mal(unsigned int count) {
  T *data = (T *)malloc(sizeof(T) * count);
  if (!data) {
    flogging(F_ERROR, "Could not malloc '" + std::to_string(sizeof(T) * count) +
                          "' bytes!");
  }
  return data;
}
extern const char *fop_to_string[];
template <typename T>
static inline std::string vectorString(const std::vector<T> &vec,
                                       std::string indentation = "") {
  std::string res = "[";
  for (size_t i = 0; i < vec.size(); i++) {
    res += std::to_string(vec[i]);
    if (i != vec.size() - 1)
      res += ", ";
  }
  return res + "]";
}
template <typename T>
static inline std::string vectorString(const std::vector<std::vector<T>> &vec,
                                       std::string indentation = "") {
  std::string res = "[";
  for (size_t i = 0; i < vec.size(); i++) {
    res += vectorString(vec[i], indentation + " ");
    if (i != vec.size() - 1)
      res += ",\n" + indentation;
  }
  return res + "]";
}
static inline int operationScore(const FGraphNode *g) {
  switch (g->operation.op_type) {
  case FPOW:
  case FSQRT:
    return 1;
  case FLOG:
  case FSIN:
  case FCOS:
  case FTAN:
  case FLOG2:
  case FLOG10:
  case FASIN:
  case FACOS:
  case FATAN:
    return 3;
  case FREDUCE_SUM:
  case FREDUCE_MUL: {
    // int dim = *((int *)g->operation.additional_data);
    // size_t total = 1;
    // for (int i = 0; i < g->operation.dimensions; i++)
    //   if (i != dim) total *= g->operation.shape[i];
    return 5; //* g->predecessors[0]->operation.shape[dim];
  }
  case FMATMUL: {
    const FGraphNode *a = g->predecessors[0];
    return 5 * a->operation.shape[a->operation.dimensions - 1];
  }
  case FGRADIENT_CONVOLVE:
  case FCONVOLVE: {
    // multiply with complete kernel size
    size_t no_elems = 1;
    const FGraphNode *a = g->predecessors[1];
    for (int i = 0; i < a->operation.dimensions; i++) {
      no_elems *= a->operation.shape[i];
    }
    return std::max(1, (int) (100 - 0.001 * no_elems * no_elems));
  }
  case FSLIDE: {
    // no multiplication with complete source, since that would artificially
    // distort parallelization
    size_t no_elems = 1;
    unsigned int *stepsize = (unsigned int *)g->operation.additional_data;
    const FGraphNode *a = g->predecessors[0];
    for (int i = 0; i < a->operation.dimensions; i++)
      no_elems *= a->operation.shape[i] /
                  (i != a->operation.dimensions - 1 ? stepsize[i] : 1);
    return (int)std::sqrt(no_elems);
  }
  case FSLICE: {
    size_t sliced_away = 1;
    const FGraphNode *p = g->predecessors[0];
    for (int i = 0; i < g->operation.dimensions; i++) {
      long diff = (p->operation.shape[i] - g->operation.shape[i]);
      if (diff > 0) sliced_away *= diff;
    }
    return sliced_away;
  }
  case FGEN_CONSTANT: {
    return 10;
  }
  default: 
    break;
  }
  return 2;
}
static inline size_t computeScore(const FGraphNode *g, bool with_pred = true) {
  std::queue<const FGraphNode *> todo;
  size_t no_elems = 1;
  for (int i = 0; i < g->operation.dimensions; i++)
    no_elems *= g->operation.shape[i];
  size_t score = 0;
  todo.push(g);
  while (!todo.empty()) {
    const FGraphNode *c = todo.front();
    todo.pop();
    score += operationScore(c);
    if (with_pred) {
      for (int i = 0; i < c->num_predecessor; i++)
        if (!c->predecessors[i]->result_data && c->operation.op_type != FSTORE)
          todo.push(c->predecessors[i]);
    }
  }
  return score * no_elems;
}
inline std::string typeString(FType t) {
  switch (t) {
  case F_INT32:
    return "int";
  case F_INT64:
    return "long";
  case F_FLOAT32:
    return "float";
  case F_FLOAT64:
    return "double";
  }
  flogging(F_ERROR, "Unknown Type: " + std::to_string((int)t));
  return "";
}
inline size_t typeSize(FType t) {
  switch (t) {
  case F_INT32:
    return sizeof(int);
  case F_INT64:
    return sizeof(long);
  case F_FLOAT32:
    return sizeof(float);
  case F_FLOAT64:
    return sizeof(double);
  }
  return 1;
}
inline FType higherType(const FType a, const FType b) {
  FType highest = F_INT32;
  if (a == F_FLOAT64 || (b == F_FLOAT64))
    highest = F_FLOAT64;
  else if (a == F_FLOAT32 || (b == F_FLOAT32))
    highest = F_FLOAT32;
  else if (a == F_INT64 || (b == F_INT64))
    highest = F_INT64;
  return highest;
}
inline std::vector<size_t> calcAccSizes(const int dimensions,
                                        const size_t *shape) {
  std::vector<size_t> acc_sizes(dimensions);
  acc_sizes[dimensions - 1] = 1;
  for (int dim = dimensions - 2; dim >= 0; dim--) {
    acc_sizes[dim] = acc_sizes[dim + 1] * shape[dim + 1];
  }
  return acc_sizes;
}
inline std::vector<size_t> calcAccSizes(const FOperation op) {
  return calcAccSizes(op.dimensions, op.shape);
}
inline std::vector<std::vector<FType>> allTypePermutations(int num) {
  using namespace std;
  if (num == 0)
    return vector<vector<FType>>{};
  if (num == 1)
    return vector<vector<FType>>{
        {F_INT32}, {F_FLOAT32}, {F_INT64}, {F_FLOAT64}};
  const vector<vector<FType>> rek = allTypePermutations(num - 1);
  vector<vector<FType>> res(rek.size() * 4);
  for (int i = 0; i < rek.size(); i++) {
    int j = 0;
    for (FType ex : {F_INT32, F_INT64, F_FLOAT32, F_FLOAT64}) {
      vector<FType> old = rek[i];
      old.push_back(ex);
      res[i * 4 + j++] = old;
    }
  }
  return res;
}
template <typename T> static constexpr FType toFlintType() {
  if (std::is_same<T, int>())
    return F_INT32;
  if (std::is_same<T, long>())
    return F_INT64;
  if (std::is_same<T, float>())
    return F_FLOAT32;
  if (std::is_same<T, double>())
    return F_FLOAT64;
  return F_INT32;
}
static std::string epsilonForType(FType type) {
  switch (type) {
  case F_FLOAT32:
    return "1.192093e-07";
  case F_FLOAT64:
    return "2.220446e-16";
  default:
    return "0";
  }
}
inline void freeAdditionalData(FGraphNode *gn) {
  switch (gn->operation.op_type) {
  case FSLICE: {
    FSlice *s = (FSlice *)gn->operation.additional_data;
    free(s->end);
    free(s->start);
    free(s->step);
    delete s;
  } break;
  case FEXTEND: {
    FExtend *s = (FExtend *)gn->operation.additional_data;
    free(s->start);
    free(s->step);
    delete s;
  } break;
  case FSLIDING_WINDOW: {
    FSlidingWindow *s = (FSlidingWindow *)gn->operation.additional_data;
    free(s->step);
    free(s->size);
    delete s;
  } break;
  case FUNSLIDE_WINDOW:
  case FGEN_ARANGE:
  case FGEN_RANDOM:
  case FGEN_CONSTANT:
  case FCONCAT:
  case FCONVOLVE:
  case FSLIDE:
  case FGRADIENT_CONVOLVE:
  case FTRANSPOSE:
  case FREDUCE_MIN:
  case FREDUCE_MAX:
  case FREDUCE_SUM:
  case FREDUCE_MUL:
    free(gn->operation.additional_data);
    gn->operation.additional_data = nullptr;
  default:
    break;
  }
}
template <typename T> class blocking_queue {
private:
  std::mutex mutex;
  std::condition_variable condition;
  std::list<T> queue;

public:
  void push_front(const T &el) {
    { // own visibility block to force destructor of lock
      std::unique_lock<std::mutex> lock(mutex);
      queue.push_front(el);
    }
    condition.notify_one();
  }
  T pop_front() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return !queue.empty(); });
    if (queue.empty()) {
#ifdef C_COMPATIBILITY
      errno = EINVAL;
      T empty;
      return empty;
#else
      throw std::runtime_error("Queue Synchronity Error!");
#endif
    }
    T foo = queue.front();
    queue.pop_front();
    return foo;
  }
};
/**
 * Generates a permutation index array for a axis of a multidimensional tensor
 * by generating for each entry in this dimension an index in the same dimension
 * with which it will be swapped.
 * The resulting permutation array is flat, has as many elements as the product
 * of shape[0] * ... * shape[ax - 1] * shape[ax] and the indices are in the
 * range between 0 and shape[ax] (so that they are only swapped inside of their
 * local dimension). Every index is referenced exactly once in its local
 * dimension.
 */
inline long *generatePermutation(size_t *shape, unsigned int ax, size_t *size) {
  size_t total_size = 1;
  for (unsigned int i = 0; i <= ax; i++)
    total_size *= shape[i];
  long *ind = safe_mal<long>(total_size);
  for (size_t k = 0; k < total_size / shape[ax]; k++) {
    const size_t base = k * shape[ax];
    for (size_t i = 0; i < shape[ax]; i++) 
      ind[base + i] = i;
    for (size_t i = 0; i < shape[ax]; i++) {
      const size_t a = base + i;
      const size_t b = base + rand() % shape[ax];
      const long v = ind[a];
      ind[a] = ind[b];
      ind[b] = v;
    }
  }
  *size = total_size;
  return ind;
}
#endif
