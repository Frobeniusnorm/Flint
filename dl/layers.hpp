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

#ifndef FLINT_LAYERS
#define FLINT_LAYERS
#include "optimizers.hpp"
#include <flint/flint.hpp>
#include <memory>
namespace LayerHelper {
/**
 * FOR INTERNAL USE ONLY
 * builds an compile-time linked list of Tensor pointer
 */
template <unsigned int index, int... w> class WeightRef {};

template <unsigned int index, int n> struct WeightRef<index, n> {
  Tensor<double, n> *weight = nullptr;
  std::unique_ptr<Optimizer> optimizer = nullptr;
  template <unsigned int k, unsigned int f>
  void set_weight(Tensor<double, f> *w) {
    if (k != index)
      flogging(F_ERROR, "Invalid weight index " + std::to_string(k) +
                            "! Only " + std::to_string(index + 1) + " exist!");
    static_assert(
        f == n,
        "Could not set weight, wrong number of dimensions! Please specify your "
        "template for the Layer super class correctly!");
    weight = w;
  }
  void gen_optimizer(const OptimizerFactory *fac) {
    optimizer = std::unique_ptr<Optimizer>(fac->generate_optimizer());
  }
};

template <unsigned int index, int n, int... wn>
struct WeightRef<index, n, wn...> {
  Tensor<double, n> *weight = nullptr;
  std::unique_ptr<Optimizer> optimizer = nullptr;
  WeightRef<index + 1, wn...> others;
  template <unsigned int k, unsigned int f>
  void set_weight(Tensor<double, f> *w) {
    if constexpr (k == index) {
      static_assert(f == n, "Could not set weight, wrong number of dimensions! "
                            "Please specify your "
                            "template for the Layer super class correctly!");
      weight = w;
    } else
      others.template set_weight<k>(w);
  }
  void gen_optimizer(const OptimizerFactory *fac) {
    optimizer = std::unique_ptr<Optimizer>(fac->generate_optimizer());
    others.gen_optimizer(fac);
  }
};
} // namespace LayerHelper
/**
 * Virtual super class to manage Layer implementations without templates
 */
struct GenericLayer {
  virtual constexpr unsigned int transform_dimensionality(unsigned int n) {
    return n;
  }
  virtual constexpr FType transform_type(FType t) {
    return t;
  }
  virtual void generate_optimizer(OptimizerFactory *factory) = 0;
};
/**
 * Virtual super class of all Layer implementations with type safe weight
 * management capabilities. The variadic template describes the dimensionality
 * of the individual weights i.e. a `Layer<3,4,5>` has three weights:
 * `Tensor<double, 3>`, `Tensor<double, 4>`, `Tensor<double, 5>`.
 */
template <int... wn> class Layer : public GenericLayer{
  LayerHelper::WeightRef<0, wn...> weight_refs;
  template <unsigned int index, unsigned int n>
  void init_weights(Tensor<double, n> *t) {
    weight_refs.template set_weight<index>(t);
  }
  template <unsigned int index, unsigned int n, typename... args>
  void init_weights(Tensor<double, n> *t, args &...weights) {
    weight_refs.template set_weight<index>(t);
    init_weights<index + 1>(weights...);
  }

public:
  Layer() = default;
  template <typename... args> Layer(args... weights) {
    init_weights<0>(weights...);
  }
  template <int index, int dim> void set_weight(Tensor<double, dim> *t) {
    weight_refs.template set_weight<index>(t);
  }
  void generate_optimizer(OptimizerFactory *factory) {
    weight_refs.gen_optimizer(factory);
  }
};
#endif
