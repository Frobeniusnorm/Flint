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

#ifndef FLINT_HPP
#define FLINT_HPP

#include "flint.h"
#include <cstring>
#include <vector>

namespace FLINT_HPP_HELPER {
template <typename T>
inline std::string vectorString(const std::vector<T> vec) {
  std::string res = "[";
  for (int i = 0; i < vec.size(); i++) {
    res += std::to_string(vec[i]);
    if (i != vec.size() - 1)
      res += ", ";
  }
  return res + "]";
}
template <typename T>
inline std::string vectorString(const std::vector<std::vector<T>> &vec) {
  std::string res = "[";
  for (int i = 0; i < vec.size(); i++) {
    res += vectorString(vec[i]);
    if (i != vec.size() - 1)
      res += ",\n";
  }
  return res + "]";
}
template <typename T>
static std::vector<T> flattened(const std::vector<std::vector<T>> vec) {
  using namespace std;
  vector<T> result;
  for (const vector<T> &v : vec) {
    result.insert(result.end(), v.begin(), v.end());
  }
  return result;
}

template <typename T>
static std::vector<T>
flattened(const std::vector<std::vector<std::vector<T>>> vec) {
  using namespace std;
  vector<T> result;
  for (const vector<vector<T>> &v : vec) {
    vector<T> rec = flattened(v);
    result.insert(result.end(), rec.begin(), rec.end());
  }
  return result;
}
}; // namespace FLINT_HPP_HELPER
template <typename T> static constexpr void isTensorType() {
  static_assert(std::is_same<T, int>() || std::is_same<T, float>() ||
                    std::is_same<T, long>() || std::is_same<T, double>(),
                "Only integer and floating-point Tensor types are allowed");
}
template <typename T> static constexpr FType toFlintType() {
  if (std::is_same<T, int>())
    return INT32;
  if (std::is_same<T, long>())
    return INT64;
  if (std::is_same<T, float>())
    return FLOAT32;
  if (std::is_same<T, double>())
    return FLOAT64;
}
template <typename K, typename V> static constexpr bool isStronger() {
  const int a = std::is_same<K, int>()     ? 0
                : std::is_same<K, long>()  ? 1
                : std::is_same<K, float>() ? 2
                                           : 3;
  const int b = std::is_same<V, int>()     ? 0
                : std::is_same<V, long>()  ? 1
                : std::is_same<V, float>() ? 2
                                           : 3;
  return a >= b;
}
template <typename T, int dimensions> struct Tensor;

// one dimensional
template <typename T> struct Tensor<T, 1> {
  template <typename K, int k> friend struct Tensor;
  typedef std::vector<T> storage_type;
  Tensor(storage_type data) : shape(data.size()) {
    isTensorType<T>();
    allocated = (T *)malloc(data.size() * sizeof(T));
    if (!allocated)
      log(ERROR, "Not enough memory!");
    std::memcpy(allocated, data.data(), data.size() * sizeof(T));
    node = createGraph(allocated, data.size(), toFlintType<T>(), &shape, 1);
  }
  // copy
  Tensor(Tensor<T, 1> &other) {
    shape = other.shape;
    node = other.node;
    storage_type other_data = *other;
    allocated = (T *)malloc(shape * sizeof(T));
    if (!allocated)
      log(ERROR, "Not enough memory!");
    memcpy(allocated, other_data.data(), shape * sizeof(T));
    node = createGraph(allocated, shape, toFlintType<T>(), &shape, 1);
  }
  void operator=(Tensor<T, 1> &other) {
    if (node)
      freeGraph(node);
    if (allocated)
      free(allocated);
    storage_type other_data = *other;
    shape = other.shape;
    std::vector<T> flat = other_data;
    allocated = (T *)malloc(flat.size() * sizeof(T));
    if (!allocated)
      log(ERROR, "Not enough memory!");
    std::memcpy(allocated, flat.data(), flat.size() * sizeof(T));
    node = createGraph(allocated, flat.size(), toFlintType<T>(), &shape, 1);
  }
  // move
  Tensor(Tensor &&other) {
    shape = other.shape;
    node = other.node;
    allocated = other.allocated;
    other.allocated = nullptr;
    other.node = nullptr;
  }
  void operator=(Tensor &&other) {
    if (node)
      freeGraph(node);
    if (allocated)
      free(allocated);
    shape = other.shape;
    node = other.node;
    allocated = other.allocated;
    other.allocated = nullptr;
    other.node = nullptr;
  }
  ~Tensor() {
    if (node)
      freeGraph(node);
    if (allocated)
      free(allocated);
  }
  std::vector<T> operator*() {
    switch (node->operation->op_type) {
    case STORE: {
      FStore *store = (FStore *)node->operation->additional_data;
      return std::vector<T>((T *)store->data,
                            (T *)store->data + store->num_entries);
    }
    case RESULTDATA: {
      FResultData *store = (FResultData *)node->operation->additional_data;
      return std::vector<T>((T *)store->data,
                            (T *)store->data + store->num_entries);
    }
    case CONST: {
      FConst *store = (FConst *)node->operation->additional_data;
      return std::vector<T>(*(T *)store->value);
    }
    default: {
      execute();
      FResultData *store = (FResultData *)node->operation->additional_data;
      return std::vector<T>((T *)store->data,
                            (T *)store->data + store->num_entries);
    }
    }
  }
  void execute() {
    FOperation *op = node->operation;
    if (op->op_type != STORE && op->op_type != RESULTDATA &&
        op->op_type != CONST)
      node = executeGraph(node);
  }
  // to calculate the return type of two tensors at compile time
  template <typename K>
  using stronger_return =
      typename std::conditional<isStronger<K, T>(), K, T>::type;
  // OPERATIONS
  template <typename K>
  Tensor<stronger_return<K>, 1> operator+(const Tensor<K, 1> other) const {
    return Tensor<stronger_return<K>, 1>(add(node, other.node), shape);
  }

protected:
  Tensor(FGraphNode *node, int shape) : node(node), shape(shape) {}
  FGraphNode *node;
  T *allocated = nullptr;
  int shape;
};

// multi dimensional
template <typename T, int n> struct Tensor {
  template <typename K, int k> friend struct Tensor;
  // storage type is the vector of the recursive type
  typedef std::vector<typename Tensor<T, n - 1>::storage_type> storage_type;
  Tensor(storage_type data) {
    isTensorType<T>();
    static_assert(n > 1, "Dimension must be at least 1");
    initShape(data);
    std::vector<T> flat = FLINT_HPP_HELPER::flattened(data);
    allocated = (T *)malloc(flat.size() * sizeof(T));
    if (!allocated)
      log(ERROR, "Not enough memory!");
    std::memcpy(allocated, flat.data(), flat.size() * sizeof(T));
    node = createGraph(allocated, flat.size(), toFlintType<T>(), shape.data(),
                       shape.size());
  }
  // copy
  Tensor(const Tensor &other) {
    std::cout << " () copy" << std::endl;
    shape = other.shape;
    const storage_type other_data = *other;
    std::vector<T> flat = FLINT_HPP_HELPER::flattened(other_data);
    allocated = (T *)malloc(flat.size() * sizeof(T));
    if (!allocated)
      log(ERROR, "Not enough memory!");
    std::memcpy(allocated, flat.data(), flat.size() * sizeof(T));
    node = createGraph(allocated, flat.size(), toFlintType<T>(), shape.data(),
                       shape.size());
  }
  void operator=(const Tensor &other) {
    std::cout << " = copy" << std::endl;
    if (node)
      freeGraph(node);
    if (allocated)
      free(allocated);
    const storage_type other_data = *other;
    shape = other.shape;
    std::vector<T> flat = FLINT_HPP_HELPER::flattened(other_data);
    allocated = (T *)malloc(flat.size() * sizeof(T));
    if (!allocated)
      log(ERROR, "Not enough memory!");
    std::memcpy(allocated, flat.data(), flat.size() * sizeof(T));
    node = createGraph(allocated, flat.size(), toFlintType<T>(), shape.data(),
                       shape.size());
  }
  // move
  Tensor(Tensor &&other) {
    std::cout << " () move" << std::endl;
    shape = other.shape;
    node = other.node;
    allocated = other.allocated;
    other.allocated = nullptr;
    other.node = nullptr;
  }
  void operator=(Tensor &&other) {
    std::cout << " = move" << std::endl;
    if (node)
      freeGraph(node);
    if (allocated)
      free(allocated);
    shape = other.shape;
    node = other.node;
    allocated = other.allocated;
    other.allocated = nullptr;
    other.node = nullptr;
  }
  ~Tensor() {
    if (node)
      freeGraph(node);
    if (allocated)
      free(allocated);
  }
  // TODO does not work
  storage_type operator*() {
    switch (node->operation->op_type) {
    case STORE: {
      FStore *store = (FStore *)node->operation->additional_data;
      storage_type result(shape[0]);
      const std::vector<T> src = std::vector<T>(
          (T *)store->data, (T *)store->data + store->num_entries);
      bringIntoShape(result, src, 0, 0, total_size);
      return result;
    }
    case RESULTDATA: {
      FResultData *store = (FResultData *)node->operation->additional_data;
      storage_type result(shape[0]);
      const std::vector<T> src = std::vector<T>(
          (T *)store->data, (T *)store->data + store->num_entries);
      bringIntoShape(result, src, 0, 0, total_size);
      return result;
    }
    default: {
      execute();
      FResultData *store = (FResultData *)node->operation->additional_data;
      storage_type result(shape[0]);
      const std::vector<T> src = std::vector<T>(
          (T *)store->data, (T *)store->data + store->num_entries);
      bringIntoShape(result, src, 0, 0, total_size);
      return result;
    }
    }
  }
  void execute() {
    FOperation *op = node->operation;
    if (op->op_type != STORE && op->op_type != RESULTDATA &&
        op->op_type != CONST)
      node = executeGraph(node);
  }
  // to calculate the return type of two tensors at compile time
  template <typename K>
  using stronger_return =
      typename std::conditional<isStronger<K, T>(), K, T>::type;
  // OPERATIONS
  template <typename K, int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  operator+(const Tensor<K, k> other) const {
    std::cout << "add: " << FLINT_HPP_HELPER::vectorString(**this) << " + "
              << FLINT_HPP_HELPER::vectorString(*other) << std::endl;
    return Tensor < stronger_return<K>,
           k >= n ? k : n > (add(node, other.node), shape);
  }

protected:
  Tensor(FGraphNode *node, std::vector<int> shape) : node(node), shape(shape) {
    total_size = 1;
    for (int ds : shape)
      total_size *= ds;
  }
  FGraphNode *node;
  T *allocated = nullptr;
  std::vector<int> shape;
  size_t total_size;
  template <typename K> void initShape(const std::vector<std::vector<K>> &vec) {
    shape.push_back(vec.size());
    if (vec.size() <= 0)
      log(ERROR, "No dimension of the Tensor may have size 0!");
    initShape(vec[0]);
    total_size *= vec.size();
  }
  template <typename K> void initShape(const std::vector<K> &vec) {
    shape.push_back(vec.size());
    total_size = vec.size();
  }
  template <typename K>
  void bringIntoShape(std::vector<K> &dest, const std::vector<K> &src, int off,
                      int dim, size_t element_size) {
    dest.insert(dest.begin(), src.begin() + off,
                src.begin() + off + shape[dim]);
  }
  template <typename K, typename V>
  void bringIntoShape(std::vector<std::vector<K>> &dest,
                      const std::vector<V> &src, int off, int dim,
                      size_t element_size) {
    int nes = element_size / shape[dim];
    for (int i = 0; i < shape[dim]; i++) {
      dest[i] = std::vector<K>(shape[dim + 1]);
      bringIntoShape(dest[i], src, off + i * nes, dim + 1, nes);
    }
  }
};
#endif
