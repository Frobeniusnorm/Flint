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

#include <vector>

template <typename T> static constexpr void isTensorType() {
  static_assert(std::is_same<T, int>() || std::is_same<T, float>() ||
                    std::is_same<T, long>() || std::is_same<T, double>(),
                "Only integer and floating-point Tensor types are allowed");
}

template <typename T, int dimensions> struct Tensor;

// one dimensional
template <typename T> struct Tensor<T, 1> {
  typedef std::vector<T> storage_type;
  Tensor() { isTensorType<T>(); }
  std::vector<T> operator*() { return data; }

private:
  std::vector<T> data;
};

// multi dimensional
template <typename T, int n> struct Tensor {
  // storage type is the vector of the recursive type
  typedef std::vector<typename Tensor<T, n - 1>::storage_type> storage_type;
  Tensor() {
    isTensorType<T>();
    static_assert(n > 1, "Dimension must be at least 1");
  }
  storage_type operator*() { return data; }

private:
  storage_type data;
};
#endif
