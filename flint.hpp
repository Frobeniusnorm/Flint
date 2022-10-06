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
#include <algorithm>
#include <array>
#include <cstring>
#include <sys/types.h>
#include <tuple>
#include <vector>
namespace FLINT_HPP_HELPER {
template <typename T>
static inline std::string vectorString(const std::vector<T> &vec) {
  std::string res = "[";
  for (size_t i = 0; i < vec.size(); i++) {
    res += std::to_string(vec[i]);
    if (i != vec.size() - 1)
      res += ", ";
  }
  return res + "]";
}
template <typename T>
static inline std::string vectorString(const std::vector<std::vector<T>> &vec) {
  std::string res = "[";
  for (size_t i = 0; i < vec.size(); i++) {
    res += vectorString(vec[i]);
    if (i != vec.size() - 1)
      res += ",\n";
  }
  return res + "]";
}
template <typename T, long unsigned int n>
static inline std::string vectorString(const std::array<T, n> &vec) {
  std::string res = "[";
  for (size_t i = 0; i < n; i++) {
    res += std::to_string(vec[i]);
    if (i != vec.size() - 1)
      res += ", ";
  }
  return res + "]";
}
template <typename T, long unsigned int n, long unsigned int k>
static inline std::string
vectorString(const std::array<std::array<T, k>, n> &vec) {
  std::string res = "[";
  for (size_t i = 0; i < n; i++) {
    res += vectorString(vec[i]);
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
    return INT32;
  if (std::is_same<T, long>())
    return INT64;
  if (std::is_same<T, float>())
    return FLOAT32;
  if (std::is_same<T, double>())
    return FLOAT64;
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
// FOR INDEXING
template <typename T, int dimensions> class TensorView;
template <typename T> class TensorView<T, 1> {
  T *data;
  const size_t already_indexed;
  const size_t shape;

public:
  TensorView(T *data, const std::vector<size_t> shape,
             const size_t already_indexed)
      : data(data), already_indexed(already_indexed), shape(shape[0]) {}
  T &operator[](size_t index) { return data[already_indexed + index]; }
  size_t size() const { return shape; }
};
template <typename T, int n> class TensorView {
  T *data;
  const size_t already_indexed;
  const std::vector<size_t> shape;

public:
  TensorView(T *data, const std::vector<size_t> shape,
             const size_t already_indexed)
      : data(data), already_indexed(already_indexed), shape(shape) {}
  TensorView<T, n - 1> operator[](size_t index) {
    std::vector<size_t> ns(shape.size() - 1);
    for (size_t i = 0; i < shape.size() - 1; i++) {
      ns[i] = shape[i + 1];
      index *= shape[i + 1];
    }
    return TensorView<T, n - 1>(data, ns, already_indexed + index);
  }
};

// FOR SLICING
struct TensorRange {
  static const size_t MAX_SIZE = -1;
  size_t start = 0;
  size_t end = MAX_SIZE;
  size_t step = 1;
  TensorRange() = default;
  TensorRange(std::tuple<size_t, size_t, size_t> range_vals)
      : start(std::get<0>(range_vals)), end(std::get<1>(range_vals)),
        step(std::get<2>(range_vals)) {}
  TensorRange(std::initializer_list<size_t> range_vals) {
    if (range_vals.size() > 0)
      start = *range_vals.begin();
    if (range_vals.size() > 1)
      end = *(range_vals.begin() + 1);
    if (range_vals.size() > 2)
      step = *(range_vals.begin() + 2);
  }
  TensorRange(size_t start, size_t end = MAX_SIZE, size_t step = 1)
      : start(start), end(end), step(step) {}
};

template <typename T, int dimensions> struct Tensor;

// one dimensional
template <typename T> struct Tensor<T, 1> {
  template <typename K, int k> friend struct Tensor;
  typedef std::vector<T> storage_type;
  typedef std::initializer_list<T> init_type;
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
  T &operator[](const size_t index) const {
    switch (node->operation->op_type) {
    case STORE: {
      FStore *store = (FStore *)node->operation->additional_data;
      return ((T *)store->data)[index];
    }
    case RESULTDATA: {
      FResultData *store = (FResultData *)node->operation->additional_data;
      return ((T *)store->data)[index];
    }
    default: {
      execute();
      FResultData *store = (FResultData *)node->operation->additional_data;
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
        op->op_type != CONST) {
      node->reference_counter--;
      node = fExecuteGraph(node);
      node->reference_counter++;
    }
  }
  operator std::string() const {
    FOperation *op = node->operation;
    std::string foo = "Tensor<" +
                      (op->data_type == INT32     ? std::string("INT32")
                       : op->data_type == INT64   ? std::string("INT64")
                       : op->data_type == FLOAT32 ? std::string("FLOAT32")
                                                  : std::string("FLOAT64")) +
                      ", shape: " + std::to_string(shape) + ">(";
    if (op->op_type != STORE && op->op_type != RESULTDATA &&
        op->op_type != CONST)
      foo += "<not yet executed>";
    else {
      switch (op->op_type) {
      case STORE: {
        FStore *store = (FStore *)node->operation->additional_data;
        foo += FLINT_HPP_HELPER::vectorString(std::vector<T>(
            (T *)store->data, (T *)store->data + store->num_entries));
        break;
      }
      case RESULTDATA: {
        FResultData *store = (FResultData *)node->operation->additional_data;
        foo += FLINT_HPP_HELPER::vectorString(std::vector<T>(
            (T *)store->data, (T *)store->data + store->num_entries));
        break;
      }
      case CONST: {
        FConst *store = (FConst *)node->operation->additional_data;
        foo += std::to_string(((T *)store->value)[0]);
        break;
      }
      default: // to make the compiler shut up
        break;
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
  template <typename K, int k>
  Tensor<stronger_return<K>, k> operator+(const Tensor<K, k> other) const {
    return Tensor<stronger_return<K>, k>(fadd(node, other.node), other.shape);
  }
  template <typename K>
  Tensor<stronger_return<K>, 1> operator+(const K con) const {
    return Tensor<stronger_return<K>, 1>(fadd(node, con), shape);
  }
  template <typename K, int k>
  Tensor<stronger_return<K>, k> operator-(const Tensor<K, k> other) const {
    return Tensor<stronger_return<K>, k>(fsub(node, other.node), other.shape);
  }
  template <typename K>
  Tensor<stronger_return<K>, 1> operator-(const K con) const {
    return Tensor<stronger_return<K>, 1>(fsub(node, con), shape);
  }
  template <typename K, int k>
  Tensor<stronger_return<K>, k> operator*(const Tensor<K, k> other) const {
    return Tensor<stronger_return<K>, k>(fmul(node, other.node), other.shape);
  }
  template <typename K>
  Tensor<stronger_return<K>, 1> operator*(const K con) const {
    return Tensor<stronger_return<K>, 1>(fmul(node, con), shape);
  }
  template <typename K, int k>
  Tensor<stronger_return<K>, k> operator/(const Tensor<K, k> other) const {
    return Tensor<stronger_return<K>, k>(fdiv(node, other.node), other.shape);
  }
  template <typename K>
  Tensor<stronger_return<K>, 1> operator/(const K con) const {
    return Tensor<stronger_return<K>, 1>(fdiv(node, con), shape);
  }
  template <typename K, int k>
  Tensor<stronger_return<K>, k> pow(const Tensor<K, k> other) const {
    return Tensor<stronger_return<K>, k>(fpow(node, other.node), other.shape);
  }
  template <typename K> Tensor<stronger_return<K>, 1> pow(const K other) const {
    return Tensor<stronger_return<K>, 1>(fpow(node, other), shape);
  }
  template <typename K, int k>
  Tensor<stronger_return<K>, k> min(const Tensor<K, k> other) const {
    return Tensor<stronger_return<K>, k>(fmin(node, other.node));
  }
  template <typename K> Tensor<stronger_return<K>, 1> min(const K other) const {
    return Tensor<stronger_return<K>, 1>(fmin(node, other));
  }
  template <typename K, int k>
  Tensor<stronger_return<K>, k> max(const Tensor<K, k> other) const {
    return Tensor<stronger_return<K>, k>(fmax(node, other.node));
  }
  template <typename K> Tensor<stronger_return<K>, 1> max(const K other) const {
    return Tensor<stronger_return<K>, 1>(fmax(node, other));
  }
  template <typename K> Tensor<K, 1> convert() const {
    return Tensor<K, 1>(fconvert(node, toFlintType<K>()), shape);
  }
  Tensor<T, 1> slice(size_t start = 0, size_t size = -1,
                     size_t step = 1) const {
    return Tensor<T, 1>(fslice_step(node, &start, &size, &step));
  }

protected:
  Tensor(FGraphNode *node, size_t shape) : node(node), shape(shape) {
    node->reference_counter++;
  }
  FGraphNode *node;
  size_t shape;
};

// multi dimensional
template <typename T, int n> struct Tensor {
  template <typename K, int k> friend struct Tensor;
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
  // copy
  Tensor(const Tensor &other) {
    shape = other.shape;
    node = fCopyGraph(other.node);
    node->reference_counter++;
  }
  void operator=(const Tensor &other) {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
    shape = other.shape;
    node = fCopyGraph(other.node);
    node->reference_counter++;
  }
  // move
  Tensor(Tensor &&other) {
    shape = other.shape;
    node = other.node; // was held by previous tensor -> no increment necessary
    other.node = nullptr;
  }
  std::array<size_t, n> get_shape() const { return shape; }
  void operator=(Tensor &&other) {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
    shape = other.shape;
    node = other.node;
    other.node = nullptr;
  }
  ~Tensor() {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
  }
  // retrieves the data of the current node and converts it into a possible
  // multidimensional vector, executes the node if necessary. The conversion has
  // to copy the complete data, because of that we recommend the builtin index
  // access of the Tensor.
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
        op->op_type != CONST) {
      // no longer held by this tensor
      node->reference_counter--;
      node = fExecuteGraph(node);
      node->reference_counter++;
    }
  }
  void execute_cpu() {
    FOperation *op = node->operation;
    if (op->op_type != STORE && op->op_type != RESULTDATA &&
        op->op_type != CONST) {
      // no longer held by this tensor
      node->reference_counter--;
      node = fExecuteGraph_cpu(node);
      node->reference_counter++;
    }
  }
  void execute_gpu() {
    FOperation *op = node->operation;
    if (op->op_type != STORE && op->op_type != RESULTDATA &&
        op->op_type != CONST) {
      // no longer held by this tensor
      node->reference_counter--;
      node = fExecuteGraph_gpu(node);
      node->reference_counter++;
    }
  }
  TensorView<T, n - 1> operator[](const size_t index) {
    switch (node->operation->op_type) {
    case STORE: {
      FStore *store = (FStore *)node->operation->additional_data;
      size_t alrdindx = index;
      std::vector<size_t> ns(shape.size() - 1);
      for (size_t i = 1; i < shape.size(); i++) {
        ns[i - 1] = shape[i];
        alrdindx *= shape[i];
      }
      return TensorView<T, n - 1>((T *)store->data, ns, alrdindx);
    }
    case RESULTDATA: {
      FResultData *store = (FResultData *)node->operation->additional_data;
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
      FResultData *store = (FResultData *)node->operation->additional_data;
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
  operator std::string() const {
    FOperation *op = node->operation;
    std::string foo = "Tensor<" +
                      (op->data_type == INT32     ? std::string("INT32")
                       : op->data_type == INT64   ? std::string("INT64")
                       : op->data_type == FLOAT32 ? std::string("FLOAT32")
                                                  : std::string("FLOAT64")) +
                      ", shape: " + FLINT_HPP_HELPER::vectorString(shape) +
                      ">(";
    if (op->op_type != STORE && op->op_type != RESULTDATA &&
        op->op_type != CONST)
      foo += "<not yet executed>";
    else {
      switch (op->op_type) {
      case STORE: {
        FStore *store = (FStore *)node->operation->additional_data;
        foo += FLINT_HPP_HELPER::vectorString(std::vector<T>(
            (T *)store->data, (T *)store->data + store->num_entries));
        break;
      }
      case RESULTDATA: {
        FResultData *store = (FResultData *)node->operation->additional_data;
        foo += FLINT_HPP_HELPER::vectorString(std::vector<T>(
            (T *)store->data, (T *)store->data + store->num_entries));
        break;
      }
      case CONST: {
        FConst *store = (FConst *)node->operation->additional_data;
        foo += std::to_string(((T *)store->value)[0]);
        break;
      }
      default:
        break;
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
  template <typename K, int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  operator+(const Tensor<K, k> other) const {
    if constexpr (k >= n)
      return Tensor<stronger_return<K>, k>(fadd(node, other.node), other.shape);
    else
      return Tensor<stronger_return<K>, n>(fadd(node, other.node), shape);
  }
  template <typename K>
  Tensor<stronger_return<K>, n> operator+(const K other) const {
    return Tensor<stronger_return<K>, n>(fadd(node, other), shape);
  }
  template <typename K, int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  operator-(const Tensor<K, k> other) const {
    if constexpr (k >= n)
      return Tensor<stronger_return<K>, k>(fsub(node, other.node), other.shape);
    else
      return Tensor<stronger_return<K>, n>(fsub(node, other.node), shape);
  }
  template <typename K>
  Tensor<stronger_return<K>, n> operator-(const K other) const {
    return Tensor<stronger_return<K>, n>(fsub(node, other), shape);
  }
  template <typename K, int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  operator*(const Tensor<K, k> other) const {
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
  template <typename K, int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  operator/(const Tensor<K, k> other) const {
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
  template <typename K, int k>
  Tensor<stronger_return<K>, n> pow(const Tensor<stronger_return<K>, k> other) {
    static_assert(
        k <= n,
        "Can't take the power of a tensor to a tensor with higher dimension!");
    return Tensor<stronger_return<K>, n>(fpow(node, other.node), shape);
  }
  template <typename K> Tensor<stronger_return<K>, n> pow(const K other) {
    return Tensor<stronger_return<K>, n>(fpow(node, other), shape);
  }
  template <typename K, int k>
  Tensor<stronger_return<K>, k >= n ? k : n> matmul(Tensor<K, k> other) {
    int x = shape[shape.size() - 2];
    int z = other.shape[other.shape.size() - 1];
    std::array<size_t, k >= n ? k : n> ns;
    for (size_t i = 0; i < ns.size() - 2; i++) {
      ns[i] = k >= n ? other.shape[i] : shape[i];
    }
    ns[ns.size() - 2] = x;
    ns[ns.size() - 1] = z;
    return Tensor < stronger_return<K>,
           k >= n ? k : n > (fmatmul(&node, &other.node), ns);
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
  template <typename K, int k>
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
  template <typename K, int k>
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
  Tensor<T, n - 1> reduce_sum(const int dimension) {
    std::array<size_t, n - 1> ns;
    for (int i = 0; i < dimension; i++)
      ns[i] = shape[i];
    for (size_t i = dimension; i < ns.size(); i++)
      ns[i] = shape[i + 1];
    return Tensor<T, n - 1>(freduce_sum(&node, dimension), ns);
  }
  Tensor<T, n - 1> reduce_mul(const int dimension) {
    std::array<size_t, n - 1> ns;
    for (int i = 0; i < dimension; i++)
      ns[i] = shape[i];
    for (size_t i = dimension; i < ns.size(); i++)
      ns[i] = shape[i + 1];
    return Tensor<T, n - 1>(freduce_mul(&node, dimension), ns);
  }
  template <typename... args> Tensor<T, n> slice(args... dim_ranges) {
    constexpr size_t num_ranges = sizeof...(args);
    static_assert(num_ranges <= n,
                  "A slice operation may only contain as many indexing ranges "
                  "as the tensor has dimensions!");
    std::array<TensorRange, num_ranges> ranges{
        static_cast<TensorRange>(dim_ranges)...};
    size_t starts[n], ends[n], steps[n];
    std::array<size_t, n> new_shape;
    for (uint i = 0; i < n; i++) {
      if (i < num_ranges) {
        starts[i] = ranges[i].start;
        ends[i] =
            ranges[i].end == TensorRange::MAX_SIZE ? shape[i] : ranges[i].end;
        steps[i] = ranges[i].step;
        new_shape[i] = ends[i] - starts[i];
        if (new_shape[i] % steps[i] == 0)
          new_shape[i] /= steps[i];
        else
          new_shape[i] = new_shape[i] / steps[i] + 1;
        // check if indexing is allowed
        if (starts[i] < 0)
          log(WARNING,
              "negative indexing leads to undefined behaviour! (start value: " +
                  std::to_string(starts[i]) + " in slice for dimension " +
                  std::to_string(i) + ")");
        if (starts[i] > ends[i])
          log(WARNING, "start index > end index yields no values for slice in "
                       "dimension " +
                           std::to_string(i) + "!");
        if (steps[i] < 1) // TODO
          log(WARNING, "currently only step sizes > 1 are supported!");
        if (ends[i] - (ends[i] % steps[i]) > shape[i])
          log(WARNING, "end index " + std::to_string(ends[i]) +
                           " in dimension " + std::to_string(i) +
                           " exceeds size of that dimension!");
      } else {
        starts[i] = 0;
        ends[i] = shape[i];
        steps[i] = 1;
        new_shape[i] = shape[i];
      }
    }
    return Tensor<T, n>(fslice_step(node, starts, ends, steps), new_shape);
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
      log(ERROR, "No dimension of the Tensor may have size 0!");
    initShape(*vec.begin(), i + 1);
    total_size *= vec.size();
  }
  template <typename K>
  void initShape(const std::initializer_list<K> &vec, int i) {
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
