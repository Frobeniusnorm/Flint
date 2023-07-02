
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
#ifndef FLINT_INITIALIZER
#define FLINT_INITIALIZER
#include <cmath>
#include <flint/flint.hpp>

template <typename T>
concept Initializer = requires(T i) {
  {
    i.template initialize<double>(std::array<size_t, 3>{1, 2, 3})
    } -> std::convertible_to<Tensor<double, 3>>;
  {
    i.template initialize<float>(std::array<size_t, 2>{6, 5})
    } -> std::convertible_to<Tensor<float, 2>>;
};
template<long unsigned int n>
static void compute_fans(std::array<size_t, n> shape, unsigned int& fan_in, unsigned int& fan_out) {
  if constexpr (n == 1) {
    fan_in = shape[0];
    fan_out = shape[0];
  } else if constexpr (n == 2) {
    fan_in = shape[0];
    fan_out = shape[1];
  } else {
    size_t acc = 1;
    for (int i = 0; i < n - 2; i++)
      acc *= shape[i];
    fan_in = shape[n-2] * acc;
    fan_out = shape[n-1] * acc;
  }
}
struct ConstantInitializer {
  double val;
  ConstantInitializer(double val = 0.0) : val(val) {}
  template <typename T, long unsigned int n>
  Tensor<T, n> initialize(std::array<size_t, n> shape) {
    return Flint::constant_array((T)val, shape);
  }
};
struct UniformRandom {
  double minval = -0.15, maxval = 0.15;
  UniformRandom(double minval = -0.15, double maxval = 0.15)
      : minval(minval), maxval(maxval) {}
  template <typename T, long unsigned int n>
  Tensor<T, n> initialize(std::array<size_t, n> shape) {
    return Flint::random_array(shape) * (maxval - minval) - minval;
  }
};

struct GlorotUniform{
  template <typename T, long unsigned int n>
  Tensor<T, n> initialize(std::array<size_t, n> shape) {
    unsigned int fan_in, fan_out;
    compute_fans(shape, fan_in, fan_out);
    double limit = std::sqrt(6. / (fan_in + fan_out));
    return Flint::random_normal(shape, limit); 
  }
};

#endif
