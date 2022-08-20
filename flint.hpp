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
#include <vector>

namespace FLINT_HPP_HELPER {
template <typename T>
static std::vector<T> flattened(std::vector<std::vector<T>> vec) {
  using namespace std;
  vector<T> result;
  for (const vector<T> &v : vec) {
    result.insert(result.end(), v.begin(), v.end());
  }
  return result;
}

template <typename T>
static std::vector<T> flattened(std::vector<std::vector<std::vector<T>>> vec) {
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
  return FType::FLOAT32;
}

template <typename T, int dimensions> struct Tensor;

// one dimensional
template <typename T> struct Tensor<T, 1> {
  typedef std::vector<T> storage_type;
  Tensor(storage_type data) : shape(data.size()) {
    isTensorType<T>();
    node = createGraph(data.data(), data.size(), toFlintType<T>(), &shape, 1);
  }
  // copy
  Tensor(const Tensor &other) {
    const storage_type other_data = *other;
    shape = other.shape;
    std::vector<T> flat = flattened(other_data);
    node = createGraph(flat.data(), flat.size(), toFlintType<T>(), &shape, 1);
  }
  void operator=(const Tensor &other) {
    if (node)
      freeGraph(node);
    const storage_type other_data = *other;
    shape = other.shape;
    std::vector<T> flat = flattened(other_data);
    node = createGraph(flat.data(), flat.size(), toFlintType<T>(), &shape, 1);
  }
  // move
  Tensor(Tensor &&other) {
    shape = other.shape;
    node = other.node;
    other.node = nullptr;
  }
  void operator=(Tensor &&other) {
    if (node)
      freeGraph(node);
    shape = other.shape;
    node = other.node;
    other.node = nullptr;
  }
  ~Tensor() { freeGraph(node); }
  std::vector<T> operator*() { return {}; }
  void execute() {}

private:
  FGraphNode *node;
  int shape;
};

// multi dimensional
template <typename T, int n> struct Tensor {
  // storage type is the vector of the recursive type
  typedef std::vector<typename Tensor<T, n - 1>::storage_type> storage_type;
  Tensor(storage_type data) : shape(data.size()) {
    isTensorType<T>();
    static_assert(n > 1, "Dimension must be at least 1");
    initShape(data, 0);
    std::vector<T> flat = FLINT_HPP_HELPER::flattened(data);
    node = createGraph(flat.data(), flat.size(), toFlintType<T>(), shape.data(),
                       shape.size());
  }
  // copy
  Tensor(const Tensor &other) {
    const storage_type other_data = *other;
    shape = other.shape;
    std::vector<T> flat = FLINT_HPP_HELPER::flattened(other_data);
    node = createGraph(flat.data(), flat.size(), toFlintType<T>(), shape.data(),
                       shape.size());
  }
  void operator=(const Tensor &other) {
    if (node)
      freeGraph(node);
    const storage_type other_data = *other;
    shape = other.shape;
    std::vector<T> flat = FLINT_HPP_HELPER::flattened(other_data);
    node = createGraph(flat.data(), flat.size(), toFlintType<T>(), shape.data(),
                       shape.size());
  }
  // move
  Tensor(Tensor &&other) {
    shape = other.shape;
    node = other.node;
    other.node = nullptr;
  }
  void operator=(Tensor &&other) {
    if (node)
      freeGraph(node);
    shape = other.shape;
    node = other.node;
    other.node = nullptr;
  }
  ~Tensor() {
    if (node)
      freeGraph(node);
  }
  storage_type operator*() { return {}; }
  void execute() {}

private:
  FGraphNode *node;
  std::vector<int> shape;

  template <typename K> void initShape(const std::vector<K> &vec, int dim) {
    shape[dim] = vec.size();
  }
  template <typename K>
  void initShape(const std::vector<std::vector<K>> &vec, int dim) {
    shape[dim] = vec.size();
    if (vec.size() <= 0)
      log(ERROR, "No dimension of the Tensor may have size 0!");
    initShape(vec[0], dim + 1);
  }
};
#endif
