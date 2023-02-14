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

/** \file flint.hpp
 * \brief This is the C++ implementation of Flint
 *
 * The core class of the C++ implementation is Tensor which has a template that
 * describes the dimensionality and type of the Tensor. All C++ functions use
 * the underlying implementations in flint.h.
 */

#include "flint.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <sys/types.h>
#include <tuple>
#include <vector>

/**
 * Useful helper functions used by the library itself.
 */
namespace FLINT_HPP_HELPER {
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
template <typename T, size_t n>
static inline std::string arrayString(const std::array<T, n> &vec) {
  std::string res = "[";
  for (size_t i = 0; i < n; i++) {
    res += std::to_string(vec[i]);
    if (i != vec.size() - 1)
      res += ", ";
  }
  return res + "]";
}
template <typename T, size_t n, size_t k>
static inline std::string
arrayString(const std::array<std::array<T, k>, n> &vec) {
  std::string res = "[";
  for (size_t i = 0; i < n; i++) {
    res += arrayString(vec[i]);
    if (i != n - 1)
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
template <typename T>
static std::vector<T>
flattened(const std::initializer_list<std::initializer_list<T>> vec) {
  using namespace std;
  vector<T> result;
  for (const initializer_list<T> &v : vec) {
    result.insert(result.end(), v.begin(), v.end());
  }
  return result;
}

template <typename T>
static std::vector<T> flattened(
    const std::initializer_list<std::initializer_list<std::initializer_list<T>>>
        vec) {
  using namespace std;
  vector<T> result;
  for (const initializer_list<initializer_list<T>> &v : vec) {
    vector<T> rec = flattened(v);
    result.insert(result.end(), rec.begin(), rec.end());
  }
  return result;
}
}; // namespace FLINT_HPP_HELPER
// checks if the given type is one of the allowed tensor types
template <typename T> static constexpr void isTensorType() {
  static_assert(std::is_same<T, int>() || std::is_same<T, float>() ||
                    std::is_same<T, long>() || std::is_same<T, double>(),
                "Only integer and floating-point Tensor types are allowed");
}
// converts c++ type to flint type
template <typename T> static constexpr FType toFlintType() {
  if (std::is_same<T, int>())
    return F_INT32;
  if (std::is_same<T, long>())
    return F_INT64;
  if (std::is_same<T, float>())
    return F_FLOAT32;
  else
    return F_FLOAT64;
}
// checks which of both types the flint backend will choose
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
/**
 * Contains static methods to configure Flints behaviour-
 */
namespace Flint {
inline void setLoggingLevel(int level) { fSetLoggingLevel(level); }
}; // namespace Flint
/**
 * Encapsulates the data of a tensor. Is only valid as long as the Tensor is
 * valid. Provides an interface for index operations on multidimensional data.
 */
template <typename T, unsigned int dimensions> class TensorView;
template <typename T> class TensorView<T, 1> {
  T *data;
  const size_t already_indexed;
  const size_t shape;

public:
  TensorView(T *data, const std::vector<size_t> shape,
             const size_t already_indexed)
      : data(data), already_indexed(already_indexed), shape(shape[0]) {}
  /**
   * Returns a read-write-reference to the index data entry of the Tensor-data.
   * Only valid as long as the original Tensor is valid.
   */
  T &operator[](size_t index) { return data[already_indexed + index]; }
  size_t size() const { return shape; }
};
template <typename T, unsigned int n> class TensorView {
  T *data;
  const size_t already_indexed;
  const std::vector<size_t> shape;

public:
  TensorView(T *data, const std::vector<size_t> shape,
             const size_t already_indexed)
      : data(data), already_indexed(already_indexed), shape(shape) {}
  /**
   * Returns a new TensorView object with one more index for the current
   * dimension (i.e. the new TensorView has one dimension less). Only valid as
   * long as the original Tensor is valid.
   */
  TensorView<T, n - 1> operator[](size_t index) {
    std::vector<size_t> ns(shape.size() - 1);
    for (size_t i = 0; i < shape.size() - 1; i++) {
      ns[i] = shape[i + 1];
      index *= shape[i + 1];
    }
    return TensorView<T, n - 1>(data, ns, already_indexed + index);
  }
};
/**
 * Describes a slice operation for one dimension.
 */
struct TensorRange {
  static const long MAX_SIZE = 2147483647;
  long start = 0;
  long end = MAX_SIZE;
  long step = 1;
  TensorRange() = default;
  TensorRange(std::tuple<long, long, long> range_vals)
      : start(std::get<0>(range_vals)), end(std::get<1>(range_vals)),
        step(std::get<2>(range_vals)) {}
  TensorRange(std::initializer_list<long> range_vals) {
    if (range_vals.size() > 0)
      start = *range_vals.begin();
    if (range_vals.size() > 1)
      end = *(range_vals.begin() + 1);
    if (range_vals.size() > 2)
      step = *(range_vals.begin() + 2);
  }
  TensorRange(long start, long end = MAX_SIZE, long step = 1)
      : start(start), end(end), step(step) {}
};

template <typename T, unsigned int dimensions> struct Tensor;

// one dimensional
template <typename T> struct Tensor<T, 1> {
  template <typename K, unsigned int k> friend struct Tensor;
  typedef std::vector<T> storage_type;
  typedef std::initializer_list<T> init_type;
  Tensor(storage_type data) : shape(data.size()) {
    isTensorType<T>();
    node = fCreateGraph(data.data(), data.size(), toFlintType<T>(), &shape, 1);
    node->reference_counter = 1;
  }
  Tensor(init_type data) : shape(data.size()) {
    isTensorType<T>();

    node = fCreateGraph(std::begin(data), data.size(), toFlintType<T>(), &shape,
                        1);
    node->reference_counter = 1;
  }
  // copy
  Tensor(const Tensor &other) {
    shape = other.shape;
    node = fCopyGraph(other.node);
    node->reference_counter++;
  }
  void operator=(const Tensor<T, 1> &other) {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
    shape = other.shape;
    node = fCopyGraph(other.node);
    node->reference_counter++;
  }
  T &operator[](const size_t index) {
    if (node->result_data)
      return ((T *)node->result_data->data)[index];
    switch (node->operation->op_type) {
    case FSTORE: {
      FStore *store = (FStore *)node->operation->additional_data;
      return ((T *)store->data)[index];
    }
    default: {
      execute();
      FResultData *store = node->result_data;
      return ((T *)store->data)[index];
    }
    }
  }
  // move
  Tensor(Tensor &&other) {
    shape = other.shape;
    node = other.node;
    other.allocated = nullptr;
    other.node = nullptr;
  }
  void operator=(Tensor &&other) {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
    shape = other.shape;
    node = other.node;
    other.allocated = nullptr;
    other.node = nullptr;
  }
  ~Tensor() {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
  }
  static Tensor<T, 1> constant(T value, size_t size) {
    FGraphNode *node = fconstant(value, &size, 1);
    return Tensor(node, size);
  }
  const size_t get_shape() const { return shape; }
  std::vector<T> operator*() {
    if (node->result_data) {
      return std::vector<T>((T *)node->result_data->data,
                            (T *)node->result_data->data +
                                node->result_data->num_entries);
    }
    switch (node->operation->op_type) {
    case FSTORE: {
      FStore *store = (FStore *)node->operation->additional_data;
      return std::vector<T>((T *)store->data,
                            (T *)store->data + store->num_entries);
    }
    case FCONST: {
      FConst *store = (FConst *)node->operation->additional_data;
      return std::vector<T>(*(T *)store->value);
    }
    default: {
      execute();
      FResultData *store = node->result_data;
      return std::vector<T>((T *)store->data,
                            (T *)store->data + store->num_entries);
    }
    }
  }
  void execute() {
    const FOperation *op = node->operation;
    if (op->op_type != FSTORE && !node->result_data && op->op_type != FCONST) {
      node->reference_counter--;
      node = fExecuteGraph(node);
      node->reference_counter++;
    }
  }
  operator std::string() const {
    FOperation *op = node->operation;
    std::string foo = "Tensor<" +
                      (op->data_type == F_INT32     ? std::string("INT32")
                       : op->data_type == F_INT64   ? std::string("INT64")
                       : op->data_type == F_FLOAT32 ? std::string("FLOAT32")
                                                    : std::string("FLOAT64")) +
                      ", shape: " + std::to_string(shape) + ">(";
    if (op->op_type != FSTORE && node->result_data && op->op_type != FCONST)
      foo += "<not yet executed>";
    else {
      if (node->result_data) {
        FResultData *store = node->result_data;
        foo += FLINT_HPP_HELPER::vectorString(std::vector<T>(
            (T *)store->data, (T *)store->data + store->num_entries));
      } else {
        switch (op->op_type) {
        case FSTORE: {
          FStore *store = (FStore *)node->operation->additional_data;
          foo += FLINT_HPP_HELPER::vectorString(std::vector<T>(
              (T *)store->data, (T *)store->data + store->num_entries));
          break;
        }
        case FCONST: {
          FConst *store = (FConst *)node->operation->additional_data;
          foo += std::to_string(((T *)store->value)[0]);
          break;
        }
        default: // to make the compiler shut up
          break;
        }
      }
    }
    foo += ")";
    return foo;
  }
  // to calculate the return type of two tensors at compile time
  template <typename K>
  using stronger_return =
      typename std::conditional<isStronger<K, T>(), K, T>::type;
  // OPERATIONS
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k> operator+(const Tensor<K, k> &other) const {
    return Tensor<stronger_return<K>, k>(fadd(node, other.node), other.shape);
  }
  template <typename K>
  Tensor<stronger_return<K>, 1> operator+(const K con) const {
    return Tensor<stronger_return<K>, 1>(fadd(node, con), shape);
  }
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k> operator-(const Tensor<K, k> &other) const {
    return Tensor<stronger_return<K>, k>(fsub(node, other.node), other.shape);
  }
  template <typename K>
  Tensor<stronger_return<K>, 1> operator-(const K con) const {
    return Tensor<stronger_return<K>, 1>(fsub(node, con), shape);
  }
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k> operator*(const Tensor<K, k> &other) const {
    return Tensor<stronger_return<K>, k>(fmul(node, other.node), other.shape);
  }
  template <typename K>
  Tensor<stronger_return<K>, 1> operator*(const K con) const {
    return Tensor<stronger_return<K>, 1>(fmul(node, con), shape);
  }
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k> operator/(const Tensor<K, k> &other) const {
    return Tensor<stronger_return<K>, k>(fdiv(node, other.node), other.shape);
  }
  template <typename K>
  Tensor<stronger_return<K>, 1> operator/(const K con) const {
    return Tensor<stronger_return<K>, 1>(fdiv(node, con), shape);
  }
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k> pow(const Tensor<K, k> &other) const {
    return Tensor<stronger_return<K>, k>(fpow(node, other.node), other.shape);
  }
  template <typename K> Tensor<stronger_return<K>, 1> pow(const K other) const {
    return Tensor<stronger_return<K>, 1>(fpow(node, other), shape);
  }
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k> min(const Tensor<K, k> &other) const {
    return Tensor<stronger_return<K>, k>(fmin(node, other.node));
  }
  template <typename K> Tensor<stronger_return<K>, 1> min(const K other) const {
    return Tensor<stronger_return<K>, 1>(fmin(node, other));
  }
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k> max(const Tensor<K, k> &other) const {
    return Tensor<stronger_return<K>, k>(fmax(node, other.node));
  }
  template <typename K> Tensor<stronger_return<K>, 1> max(const K other) const {
    return Tensor<stronger_return<K>, 1>(fmax(node, other));
  }
  template <typename K> Tensor<K, 1> convert() const {
    return Tensor<K, 1>(fconvert(node, toFlintType<K>()), shape);
  }
  Tensor<T, 1> abs() const { return Tensor<T, 1>(fabs_g(node), shape); }
  Tensor<T, 1> slice(long start = 0, long end = TensorRange::MAX_SIZE,
                     long step = 1) const {
    if (start == TensorRange::MAX_SIZE)
      start = shape - 1;
    if (end == TensorRange::MAX_SIZE)
      end = shape;
    FGraphNode *nn = fslice_step(node, &start, &end, &step);
    return Tensor<T, 1>(nn, nn->operation->shape[0]);
  }
  FGraphNode *get_graph_node() const { return node; }

  Tensor<T, 1> repeat(int repetitions) const {
    FGraphNode *nn = frepeat(node, &repetitions);
    return Tensor<T, 1>(nn, (shape * repetitions + 1));
  }
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k> gradient(const Tensor<K, k> &dx) {
    return Tensor<stronger_return<K>, k>(
        fCalculateGradient(this->node, dx.node), dx.shape);
  }

protected:
  Tensor(FGraphNode *node, size_t shape) : node(node), shape(shape) {
    node->reference_counter++;
  }
  FGraphNode *node;
  size_t shape;
};

// multi dimensional
template <typename T, unsigned int n> struct Tensor {
  template <typename K, unsigned int k> friend struct Tensor;
  // storage type is the vector of the recursive type
  typedef std::vector<typename Tensor<T, n - 1>::storage_type> storage_type;
  typedef std::initializer_list<typename Tensor<T, n - 1>::init_type> init_type;
  Tensor(init_type data) {
    isTensorType<T>();
    static_assert(n > 1, "Dimension must be at least 1");
    initShape(data, 0);
    std::vector<T> flat = FLINT_HPP_HELPER::flattened(data);
    node = fCreateGraph(flat.data(), flat.size(), toFlintType<T>(),
                        shape.data(), shape.size());
    // the node which is currently hold is always referenced
    node->reference_counter = 1;
  }
  Tensor(storage_type data) {
    isTensorType<T>();
    static_assert(n > 1, "Dimension must be at least 1");
    initShape(data, 0);
    std::vector<T> flat = FLINT_HPP_HELPER::flattened(data);
    node = fCreateGraph(flat.data(), flat.size(), toFlintType<T>(),
                        shape.data(), shape.size());
    // the node which is currently hold is always referenced
    node->reference_counter = 1;
  }
  // copy
  Tensor(const Tensor &other) {
    shape = other.shape;
    total_size = other.total_size;
    node = fCopyGraph(other.node);
    node->reference_counter++;
  }
  Tensor<T, n> &operator=(const Tensor &other) {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
    shape = other.shape;
    total_size = other.total_size;
    node = fCopyGraph(other.node);
    node->reference_counter++;
    return *this;
  }
  // move
  Tensor(Tensor &&other) {
    shape = other.shape;
    total_size = other.total_size;
    node = other.node; // was held by previous tensor -> no increment necessary
    other.node = nullptr;
  }
  std::array<size_t, n> get_shape() const { return shape; }
  Tensor<T, n> &operator=(Tensor &&other) {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
    shape = other.shape;
    total_size = other.total_size;
    node = other.node;
    other.node = nullptr;
    return *this;
  }
  ~Tensor() {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
  }
  template <typename... args>
  static Tensor<T, n> constant(T value, args... sizes) {
    constexpr size_t dimensions = sizeof...(args);
    std::array<size_t, dimensions> shape{static_cast<size_t>(sizes)...};
    FGraphNode *node = fconstant(value, shape.data(), dimensions);
    return Tensor(node, shape);
  }
  // retrieves the data of the current node and converts it into a possible
  // multidimensional vector, executes the node if necessary. The conversion has
  // to copy the complete data, because of that we recommend the builtin index
  // access of the Tensor.
  storage_type operator*() {
    if (node->result_data) {
      FResultData *store = node->result_data;
      storage_type result(shape[0]);
      const std::vector<T> src = std::vector<T>(
          (T *)store->data, (T *)store->data + store->num_entries);
      bringIntoShape(result, src, 0, 0, total_size);
      return result;
    }
    switch (node->operation->op_type) {
    case FSTORE: {
      FStore *store = (FStore *)node->operation->additional_data;
      storage_type result(shape[0]);
      const std::vector<T> src = std::vector<T>(
          (T *)store->data, (T *)store->data + store->num_entries);
      bringIntoShape(result, src, 0, 0, total_size);
      return result;
    }
    default: {
      execute();
      FResultData *store = node->result_data;
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
    if (op->op_type != FSTORE && !node->result_data && op->op_type != FCONST) {
      node = fExecuteGraph(node);
    }
  }
  void execute_cpu() {
    FOperation *op = node->operation;
    if (op->op_type != FSTORE && !node->result_data && op->op_type != FCONST) {
      node = fExecuteGraph_cpu(node);
    }
  }
  void execute_gpu() {
    FOperation *op = node->operation;
    if (op->op_type != FSTORE && !node->result_data && op->op_type != FCONST) {
      node = fExecuteGraph_gpu(node);
    }
  }
  TensorView<T, n - 1> operator[](const size_t index) {
    if (node->result_data) {
      FResultData *store = node->result_data;
      size_t alrdindx = index;
      std::vector<size_t> ns(shape.size() - 1);
      for (size_t i = 1; i < shape.size(); i++) {
        ns[i - 1] = shape[i];
        alrdindx *= shape[i];
      }
      return TensorView<T, n - 1>((T *)store->data, ns, alrdindx);
    }
    switch (node->operation->op_type) {
    case FSTORE: {
      FStore *store = (FStore *)node->operation->additional_data;
      size_t alrdindx = index;
      std::vector<size_t> ns(shape.size() - 1);
      for (size_t i = 1; i < shape.size(); i++) {
        ns[i - 1] = shape[i];
        alrdindx *= shape[i];
      }
      return TensorView<T, n - 1>((T *)store->data, ns, alrdindx);
    }
    default: {
      execute();
      FResultData *store = node->result_data;
      size_t alrdindx = index;
      std::vector<size_t> ns(shape.size() - 1);
      for (size_t i = 1; i < shape.size(); i++) {
        ns[i - 1] = shape[i];
        alrdindx *= shape[i];
      }
      return TensorView<T, n - 1>((T *)store->data, ns, alrdindx);
    }
    }
  }
  operator std::string() {
    FOperation *op = node->operation;
    std::string foo = "Tensor<" +
                      (op->data_type == F_INT32     ? std::string("INT32")
                       : op->data_type == F_INT64   ? std::string("INT64")
                       : op->data_type == F_FLOAT32 ? std::string("FLOAT32")
                                                    : std::string("FLOAT64")) +
                      ", shape: " + FLINT_HPP_HELPER::arrayString(shape) + ">(";
    if (op->op_type != FSTORE && !node->result_data && op->op_type != FCONST)
      foo += "<not yet executed>";
    else {
      foo += "\n" + FLINT_HPP_HELPER::vectorString(this->operator*(), " ");
    }
    foo += ")";
    return foo;
  }
  // to calculate the return type of two tensors at compile time
  template <typename K>
  using stronger_return =
      typename std::conditional<isStronger<K, T>(), K, T>::type;

  // OPERATIONS
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  operator+(const Tensor<K, k> &other) const {
    if constexpr (k >= n)
      return Tensor<stronger_return<K>, k>(fadd(node, other.node), other.shape);
    else
      return Tensor<stronger_return<K>, n>(fadd(node, other.node), shape);
  }
  template <typename K>
  Tensor<stronger_return<K>, n> operator+(const K other) const {
    return Tensor<stronger_return<K>, n>(fadd(node, other), shape);
  }
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  operator-(const Tensor<K, k> &other) const {
    if constexpr (k >= n)
      return Tensor<stronger_return<K>, k>(fsub(node, other.node), other.shape);
    else
      return Tensor<stronger_return<K>, n>(fsub(node, other.node), shape);
  }
  template <typename K>
  Tensor<stronger_return<K>, n> operator-(const K other) const {
    return Tensor<stronger_return<K>, n>(fsub(node, other), shape);
  }
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  operator*(const Tensor<K, k> &other) const {
    if constexpr (k >= n)
      return Tensor<stronger_return<K>, k>(fmul(node, other.node), other.shape);
    else
      return Tensor<stronger_return<K>, n>(fmul(node, other.node), shape);
  }
  template <typename K>
  Tensor<stronger_return<K>, n> operator*(const K other) const {
    return Tensor<stronger_return<K>, n>(fmul(node, other), shape);
  }
  Tensor<T, n> operator-() const { return Tensor<T, n>(fmul(node, -1), shape); }
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  operator/(const Tensor<K, k> &other) const {
    if constexpr (k >= n)
      return Tensor<stronger_return<K>, k>(fdiv(node, other.node), other.shape);
    else
      return Tensor<stronger_return<K>, n>(fdiv(node, other.node), shape);
  }
  template <typename K>
  Tensor<stronger_return<K>, n> operator/(const K other) const {
    return Tensor<stronger_return<K>, n>(fdiv(node, other), shape);
  }
  Tensor<T, 1> flattened() const {
    FGraphNode *foo = fflatten(node);
    return Tensor<T, 1>(foo, total_size);
  }
  Tensor<T, n - 1> flattened(const int dimension) const {
    FGraphNode *foo = fflatten_dimension(node, dimension);
    std::array<size_t, n - 1> ns;
    std::copy_n(foo->operation->shape, (size_t)foo->operation->dimensions,
                ns.begin());
    return Tensor<T, n - 1>(foo, ns);
  }
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, n>
  pow(const Tensor<stronger_return<K>, k> &other) {
    static_assert(
        k <= n,
        "Can't take the power of a tensor to a tensor with higher dimension!");
    return Tensor<stronger_return<K>, n>(fpow(node, other.node), shape);
  }
  template <typename K> Tensor<stronger_return<K>, n> pow(const K other) {
    return Tensor<stronger_return<K>, n>(fpow(node, other), shape);
  }
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k >= n ? k : n> matmul(Tensor<K, k> &other) {
    int x = shape[shape.size() - 2];
    int z = other.shape[other.shape.size() - 1];
    std::array<size_t, k >= n ? k : n> ns;
    for (size_t i = 0; i < ns.size() - 2; i++) {
      ns[i] = k >= n ? other.shape[i] : shape[i];
    }
    ns[ns.size() - 2] = x;
    ns[ns.size() - 1] = z;
    return Tensor < stronger_return<K>,
           k >= n ? k : n > (fmatmul(node, other.node), ns);
  }
  template <typename K> Tensor<K, n> convert() const {
    return Tensor<K, n>(fconvert(node, toFlintType<K>()), shape);
  }
  template <typename... args>
  Tensor<T, sizeof...(args)> reshape(args... shape) {
    constexpr size_t newdim = sizeof...(args);
    std::array<size_t, newdim> new_shape{static_cast<size_t>(shape)...};
    return Tensor<T, newdim>(freshape(node, new_shape.data(), newdim),
                             new_shape);
  }
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  min(const Tensor<K, k> other) const {
    if constexpr (k >= n)
      return Tensor<stronger_return<K>, k>(fmin(node, other.node), other.shape);
    else
      return Tensor<stronger_return<K>, n>(fmin(node, other.node), shape);
  }
  template <typename K> Tensor<stronger_return<K>, n> min(const K other) const {
    return Tensor<stronger_return<K>, n>(fmin(node, other));
  }
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  max(const Tensor<K, k> other) const {
    if constexpr (k >= n)
      return Tensor<stronger_return<K>, k>(fmax(node, other.node), other.shape);
    else
      return Tensor<stronger_return<K>, n>(fmax(node, other.node), shape);
  }
  template <typename K> Tensor<stronger_return<K>, n> max(const K other) const {
    return Tensor<stronger_return<K>, n>(fmax(node, other));
  }
  Tensor<T, n - 1> reduce_sum(int dimension) {
    std::array<size_t, n - 1> ns;
    if (dimension < 0)
      dimension = shape.size() + dimension;
    for (int i = 0; i < dimension; i++)
      ns[i] = shape[i];
    for (size_t i = dimension; i < ns.size(); i++)
      ns[i] = shape[i + 1];
    return Tensor<T, n - 1>(freduce_sum(node, dimension), ns);
  }
  Tensor<T, n - 1> reduce_mul(int dimension) {
    std::array<size_t, n - 1> ns;
    if (dimension < 0)
      dimension = shape.size() + dimension;
    for (int i = 0; i < dimension; i++)
      ns[i] = shape[i];
    for (size_t i = dimension; i < ns.size(); i++)
      ns[i] = shape[i + 1];
    return Tensor<T, n - 1>(freduce_mul(node, dimension), ns);
  }
  Tensor<T, n> abs() const { return Tensor<T, n>(fabs_g(node), shape); }

  template <typename... args>
  Tensor<T, n> slice(const args... dim_ranges) const {
    constexpr size_t num_ranges = sizeof...(args);
    static_assert(num_ranges <= n,
                  "A slice operation may only contain as many indexing ranges "
                  "as the tensor has dimensions!");
    std::array<TensorRange, num_ranges> ranges{
        static_cast<TensorRange>(dim_ranges)...};
    long starts[n], ends[n], steps[n];
    for (unsigned int i = 0; i < n; i++) {
      if (i < num_ranges) {
        starts[i] = ranges[i].start == TensorRange::MAX_SIZE ? shape[i] - 1
                                                             : ranges[i].start;
        ends[i] =
            ranges[i].end == TensorRange::MAX_SIZE ? shape[i] : ranges[i].end;
        steps[i] = ranges[i].step;
      } else {
        starts[i] = 0;
        ends[i] = shape[i];
        steps[i] = 1;
      }
    }
    FGraphNode *nn = fslice_step(node, starts, ends, steps);
    std::array<size_t, n> new_shape;
    for (size_t i = 0; i < n; i++)
      new_shape[i] = nn->operation->shape[i];
    return Tensor<T, n>(nn, new_shape);
  }

  template <typename... args>
  Tensor<T, n> repeat(const args... repetitions) const {
    constexpr size_t num_repetitions = sizeof...(args);
    static_assert(num_repetitions <= n,
                  "A repetition operation may can only as many repetition "
                  "entries as there are dimensions in the tensor!");
    std::array<int, num_repetitions> repeat{static_cast<int>(repetitions)...};
    std::array<int, n> acc_repeat;
    for (int i = 0; i < num_repetitions; i++)
      acc_repeat[i] = repeat[i];
    for (int i = num_repetitions; i < n; i++)
      acc_repeat[i] = 0;
    FGraphNode *nn = frepeat(node, acc_repeat.data());
    std::array<size_t, n> new_shape;
    for (size_t i = 0; i < n; i++)
      new_shape[i] = nn->operation->shape[i];
    return Tensor<T, n>(nn, new_shape);
  }
  Tensor<T, n> transpose(std::initializer_list<int> transposition = {}) {
    std::array<int, n> acc_trans;
    int i = 0;
    for (auto it = transposition.begin(); it < transposition.end(); it++)
      acc_trans[i++] = *it;
    while (i < n)
      acc_trans[i++] = n - i - 1;
    std::array<size_t, n> new_shape;
    for (int j = 0; j < n; j++)
      new_shape[j] = shape[acc_trans[j]];
    FGraphNode *nn = ftranspose(node, acc_trans.data());
    return Tensor<T, n>(nn, new_shape);
  }
  FGraphNode *get_graph_node() const { return node; }

  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k> gradient(const Tensor<K, k> &dx) {
    return Tensor<stronger_return<K>, k>(
        fCalculateGradient(this->node, dx.node), dx.shape);
  }

protected:
  Tensor(FGraphNode *node, std::array<size_t, n> shape)
      : node(node), shape(shape) {
    // here data is still fine
    total_size = 1;
    for (int ds : shape)
      total_size *= ds;
    node->reference_counter++;
  }
  FGraphNode *node;
  std::array<size_t, n> shape;
  size_t total_size;
  template <typename K>
  void initShape(const std::initializer_list<std::initializer_list<K>> &vec,
                 int i) {
    shape[i] = vec.size();
    if (vec.size() <= 0)
      flogging(F_ERROR, "No dimension of the Tensor may have size 0!");
    initShape(*vec.begin(), i + 1);
    total_size *= vec.size();
  }
  template <typename K>
  void initShape(const std::initializer_list<K> &vec, int i) {
    shape[i] = vec.size();
    total_size = vec.size();
  }
  template <typename K>
  void initShape(const std::vector<std::vector<K>> &vec, int i) {
    shape[i] = vec.size();
    if (vec.size() <= 0)
      flogging(F_ERROR, "No dimension of the Tensor may have size 0!");
    initShape(*vec.begin(), i + 1);
    total_size *= vec.size();
  }
  template <typename K> void initShape(const std::vector<K> &vec, int i) {
    shape[i] = vec.size();
    total_size = vec.size();
  }
  template <typename K>
  void bringIntoShape(std::vector<K> &dest, const std::vector<K> &src, int off,
                      int dim, size_t element_size) {
    for (size_t i = 0; i < shape[dim]; i++) {
      dest[i] = src[off + i];
    }
  }
  template <typename K, typename V>
  void bringIntoShape(std::vector<std::vector<K>> &dest,
                      const std::vector<V> &src, int off, int dim,
                      size_t element_size) {
    int nes = element_size / shape[dim];
    for (size_t i = 0; i < shape[dim]; i++) {
      dest[i] = std::vector<K>(shape[dim + 1]);
      bringIntoShape(dest[i], src, off + i * nes, dim + 1, nes);
    }
  }
};
#endif
