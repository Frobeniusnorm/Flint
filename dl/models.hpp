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

template <FType in, GenericLayer  K> constexpr FType get_output_type() {
  return K::transform_type(in);
}
template <FType in, GenericLayer K, GenericLayer... F>
constexpr FType get_output_type() {
  constexpr FType out = K::transform_type(in);
  return get_output_type<out, F...>();
}

template <unsigned int in, GenericLayer K> constexpr unsigned int get_output_dim() {
  return K::transform_dim(in);
}
template <unsigned int in, GenericLayer K, GenericLayer... F> constexpr unsigned int get_output_dim() {
  constexpr unsigned int out = K::transform_dim(in);
  return get_output_dim<out, F...>();
}

template <GenericLayer... T> struct SequentialModel {
  std::tuple<T...> layers;
  SequentialModel(T... layers) : layers(std::move(layers)...) {
  }
  void generate_optimizer(OptimizerFactory *fac) { gen_opt<0>(fac); }

  template <typename K, unsigned int n>
  Tensor<LayerHelper::FlintTypeToCpp<get_output_type<toFlintType<K>(), T...>()>,
         get_output_dim<n, T...>()>
  forward(Tensor<K, n> &in) {
    FGraphNode *out = forward(in);
    constexpr unsigned int out_dim = get_output_dim<n, T...>();
    return Tensor<
        LayerHelper::FlintTypeToCpp<get_output_type<toFlintType<K>(), T...>()>,
        out_dim>(out,
                 std::array<size_t, out_dim>(out->operation->shape,
                                             out->operation->shape + out_dim));
  }

private:
  template <int n> void gen_opt(OptimizerFactory *fac) {
    if constexpr (n < sizeof...(T)) {
      std::get<n>(layers).generate_optimizer(fac);
      gen_opt<n + 1>(fac);
    }
  }
  template <int n> FGraphNode *forward(FGraphNode *in) {
    if constexpr (n < sizeof...(T)) {
      FGraphNode *out = std::get<n>(layers).forward(in);
      return forward<n + 1>(out);
    }
    return in;
  }
};
#endif
