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
#include <concepts>
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
  template <typename T, unsigned int k>
  void optimize(const Tensor<T, k>& error) {
    if (optimizer && weight) {
      const Tensor<double, n> gw = error.gradient(*weight);
      FGraphNode* new_graph_node = optimizer->update(weight->get_graph_node(), gw.get_graph_node());
      (*weight) = Tensor<double, n>(new_graph_node, weight->get_shape());
    }
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
  template <typename T, unsigned int k>
  void optimize(const Tensor<T, k>& error) {
    if (optimizer && weight) {
      const Tensor<double, n> gw = error.gradient(*weight);
      FGraphNode* new_graph_node = optimizer->update(weight->get_graph_node(), gw.get_graph_node());
      (*weight) = Tensor<double, n>(new_graph_node, weight->get_shape());
    }
    others.optimize(error);
  }
};
template <FType t>
using FlintTypeToCpp = typename std::conditional<
    t == F_INT32, int,
    typename std::conditional<
        t == F_INT64, long,
        typename std::conditional<t == F_FLOAT32, float, double>::type>::type>::
    type;
} // namespace LayerHelper
template <unsigned int> using helper = void;
template <typename T>
concept GenericLayer =
    requires(T a, Tensor<float, 2> &t1, Tensor<int, 2> &t2,
             Tensor<double, 2> &t3, Tensor<long, 2> &t4,
             OptimizerFactory *fac) {
      {
        a.forward(t1)
        } -> std::convertible_to<
            Tensor<LayerHelper::FlintTypeToCpp<T::transform_type(F_FLOAT32)>,
                   T::transform_dimensionality(2)>>;
      {
        a.forward(t2)
        } -> std::convertible_to<
            Tensor<LayerHelper::FlintTypeToCpp<T::transform_type(F_INT32)>,
                   T::transform_dimensionality(2)>>;
      {
        a.forward(t3)
        } -> std::convertible_to<
            Tensor<LayerHelper::FlintTypeToCpp<T::transform_type(F_FLOAT64)>,
                   T::transform_dimensionality(2)>>;
      {
        a.forward(t4)
        } -> std::convertible_to<
            Tensor<LayerHelper::FlintTypeToCpp<T::transform_type(F_INT64)>,
                   T::transform_dimensionality(2)>>;
      a.optimize_weights(t1);
      a.optimize_weights(t2);
      a.optimize_weights(t3);
      a.optimize_weights(t4);
      a.generate_optimizer(fac);
      { T::transform_dimensionality(5) } -> std::convertible_to<unsigned int>;
      // Has to be constexpr

      { T::transform_type(F_INT32) } -> std::convertible_to<FType>;
      // Has to be constexpr
    };
//    };
/**
 * Virtual super class of all Layer implementations with type safe weight
 * management capabilities. The variadic template describes the dimensionality
 * of the individual weights i.e. a `Layer<3,4,5>` has three weights:
 * `Tensor<double, 3>`, `Tensor<double, 4>`, `Tensor<double, 5>`.
 */
template <int... wn> class Layer {
protected:
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
  static constexpr FType transform_type(FType t) { return F_FLOAT64; }
  static constexpr unsigned int transform_dimensionality(unsigned int n) {
    return n;
  }
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
  template <typename T, unsigned int dim>
  void optimize_weights(const Tensor<T, dim>& error) {
    weight_refs.optimize(error);
  }
};

struct Connected : public Layer<2> {
  static constexpr FType transform_type(FType t) { return F_FLOAT64; }
  Tensor<double, 2> weights;

  Connected(size_t units_in, size_t units_out)
      : Layer(&weights),
        weights(Flint::random(units_in + 1, units_out)) {}

  template <typename T, unsigned int n>
  Tensor<double, n> forward(Tensor<T, n> &in) {
    std::array<size_t, n> one_shape = in.get_shape();
    one_shape[n - 1] = 1;
    Tensor<T, n> ones = Flint::constant<T, n>(1, one_shape);
    return Flint::concat(in, ones, n - 1).matmul(weights);
  }
};
#endif
