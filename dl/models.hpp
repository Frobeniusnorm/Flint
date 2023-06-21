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
#ifndef FLINT_MODELS
#define FLINT_MODELS
#include "layers.hpp"
#include "optimizers.hpp"
#include <flint/flint.h>
#include <flint/flint_helper.hpp>
#include <memory>
#include <tuple>
#include <vector>

template <FType in> constexpr FType get_output_type() {
  return in;
}
template <FType in, GenericLayer K> constexpr FType get_output_type() {
  return K::transform_type(in);
}
template <FType in, GenericLayer K1, GenericLayer K2, GenericLayer... F>
constexpr FType get_output_type() {
  constexpr FType out = K2::transform_type(K1::transform_type(in));
  return get_output_type<out, F...>();
}
template <unsigned int in>
constexpr unsigned int get_output_dim() {
  return in;
}
template <unsigned int in, GenericLayer K>
constexpr unsigned int get_output_dim() {
  return K::transform_dimensionality(in);
}
template <unsigned int in, GenericLayer K1, GenericLayer K2, GenericLayer... F>
constexpr unsigned int get_output_dim() {
  constexpr unsigned int out = K2::transform_dimensionality(K1::transform_dimensionality(in));
  return get_output_dim<out, F...>();
}

template <GenericLayer... T> struct SequentialModel {
  std::tuple<T...> layers;
  SequentialModel(T... layers) : layers(std::move(layers)...) {}
  void generate_optimizer(OptimizerFactory *fac) { gen_opt<0>(fac); }

  template <typename K, unsigned int n>
  Tensor<LayerHelper::FlintTypeToCpp<get_output_type<toFlintType<K>(), T...>()>,
         get_output_dim<n, T...>()>
  forward(Tensor<K, n> &in) {
    return forward_helper<0, LayerHelper::FlintTypeToCpp<get_output_type<toFlintType<K>(), T...>()>,
         get_output_dim<n, T...>()>(in);
  }
  template <typename K, unsigned int n>
  void optimize(const Tensor<K, n> &error) {}

private:
  template <int n, int k, typename K> void backward(const Tensor<K, k> &error) {
    if constexpr (n < sizeof...(T)) {
      std::get<n>(layers).optimize_weights(error);
      backward(error);
    }
  }
  template <int n> void gen_opt(OptimizerFactory *fac) {
    if constexpr (n < sizeof...(T)) {
      std::get<n>(layers).generate_optimizer(fac);
      gen_opt<n + 1>(fac);
    }
  }
  template <int layer, typename T2, unsigned int n2, typename T1,
            unsigned int n1>
  Tensor<T2, n2> forward_helper(Tensor<T1, n1> &in) {
    auto out = std::get<layer>(layers).forward(in);
    if constexpr (layer == sizeof...(T) - 1)
      return out;
    else
      return forward_helper<layer + 1, T2, n2>(out);
  }
};
#endif
