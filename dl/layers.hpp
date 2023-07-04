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
#include "initializer.hpp"
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
  Tensor<double, n> weight;
  std::unique_ptr<Optimizer> optimizer = nullptr;
  template <unsigned int k, unsigned int f>
  void set_weight(Tensor<double, f> &w) {
    static_assert(k == index, "Invalid weight index!");
    static_assert(
        f == n,
        "Could not set weight, wrong number of dimensions! Please specify your "
        "template for the Layer super class correctly!");
    weight = w;
    weight.watch();
  }
  void gen_optimizer(const OptimizerFactory *fac) {
    optimizer = std::unique_ptr<Optimizer>(fac->generate_optimizer());
  }
  template <typename T, unsigned int k>
  void optimize(const Tensor<T, k> &error) {
    if (optimizer) {
      const Tensor<double, n> gw = error.gradient(weight); // this line - somehow - is the problem
      FGraphNode *new_graph_node =
          optimizer->update(weight.get_graph_node(), gw.get_graph_node());
      Tensor<double, n> nw =
          Tensor<double, n>(new_graph_node, weight.get_shape());
      weight = std::move(nw);
      weight.watch();
    } else {
      flogging(F_WARNING, "No Optimizer for weight!");
    }
  }
  template <int i, unsigned int k> Tensor<double, k> &get_weight() {
    static_assert(i == index, "Invalid weight index!");
    return weight;
  }
};

template <unsigned int index, int n, int... wn>
struct WeightRef<index, n, wn...> {
  Tensor<double, n> weight;
  std::unique_ptr<Optimizer> optimizer = nullptr;
  WeightRef<index + 1, wn...> others;
  template <unsigned int k, unsigned int f>
  void set_weight(Tensor<double, f> &w) {
    if constexpr (k == index) {
      static_assert(f == n, "Could not set weight, wrong number of dimensions! "
                            "Please specify your "
                            "template for the Layer super class correctly!");
      weight = w;
      weight.watch();
    } else
      others.template set_weight<k>(w);
  }
  void gen_optimizer(const OptimizerFactory *fac) {
    optimizer = std::unique_ptr<Optimizer>(fac->generate_optimizer());
    others.gen_optimizer(fac);
  }
  template <typename T, unsigned int k>
  void optimize(const Tensor<T, k> &error) {
    if (optimizer) {
      const Tensor<double, n> gw = error.gradient(weight);
      FGraphNode *new_graph_node =
          optimizer->update(weight.get_graph_node(), gw.get_graph_node());
      Tensor<double, n> nw =
          Tensor<double, n>(new_graph_node, weight.get_shape());
      weight = std::move(nw);
      weight.watch();
    } else {
      flogging(F_WARNING, "No Optimizer for weight!");
    }
    others.optimize(error);
  }
  template <int i, unsigned int k> Tensor<double, k> &get_weight() {
    if (i == index)
      return weight;
    else
      others.template get_weight<i, k>();
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
concept GenericLayer = requires(T a, Tensor<float, 2> &t1, Tensor<int, 2> &t2,
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
  a.training = true;
  { T::transform_dimensionality(5) } -> std::convertible_to<unsigned int>;
  // Has to be constexpr

  { T::transform_type(F_INT32) } -> std::convertible_to<FType>;
  // Has to be constexpr
};
//    };
/** Implements blank methods for every method of GenericLayer that is not needed
 * for a Layer that is not trainable */
struct UntrainableLayer {
  bool training = false;
  // to fulfill generic layer
  void generate_optimizer(OptimizerFactory *factory) {}
  // to fulfill generic layer
  template <typename T, unsigned int dim>
  void optimize_weights(const Tensor<T, dim> &error) {}
  static constexpr FType transform_type(FType t) { return t; }
  static constexpr unsigned int transform_dimensionality(unsigned int n) {
    return n;
  }
};
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
  void init_weights(Tensor<double, n> &t) {
    weight_refs.template set_weight<index>(t);
  }
  template <unsigned int index, unsigned int n, typename... args>
  void init_weights(Tensor<double, n> &t, args &...weights) {
    weight_refs.template set_weight<index>(t);
    init_weights<index + 1>(weights...);
  }
  template <int index, int w, int... wo> static constexpr int get_dim() {
    if constexpr (index == 0)
      return w;
    else
      return get_dim<index - 1, wo...>();
  }

public:
  bool training = false;
  static constexpr FType transform_type(FType t) { return F_FLOAT64; }
  static constexpr unsigned int transform_dimensionality(unsigned int n) {
    return n;
  }
  Layer() = default;
  template <typename... args> Layer(args... weights) {
    init_weights<0>(weights...);
  }
  template <int index, int dim> void set_weight(Tensor<double, dim> t) {
    weight_refs.template set_weight<index>(std::move(t));
  }
  template <int index> Tensor<double, get_dim<index, wn...>()> &get_weight() {
    return weight_refs.template get_weight<index, get_dim<index, wn...>()>();
  }
  void generate_optimizer(OptimizerFactory *factory) {
    weight_refs.gen_optimizer(factory);
  }
  template <typename T, unsigned int dim>
  void optimize_weights(const Tensor<T, dim> &error) {
    weight_refs.optimize(error);
  }
};

struct Connected : public Layer<2> {
  template <Initializer InitWeights, Initializer InitBias>
  Connected(size_t units_in, size_t units_out, InitWeights init_weights,
            InitBias init_bias)
      : Layer<2>(Flint::concat(init_weights.template initialize<double>(
                                   std::array<size_t, 2>{units_in, units_out}),
                               init_bias.template initialize<double>(
                                   std::array<size_t, 2>{1, units_out}),
                               0)) {}
  Connected(size_t units_in, size_t units_out)
      : Layer<2>(
            Flint::concat(GlorotUniform().template initialize<double>(
                              std::array<size_t, 2>{units_in, units_out}),
                          ConstantInitializer().template initialize<double>(
                              std::array<size_t, 2>{1, units_out}),
                          0)) {}
  template <typename T, unsigned int n>
  Tensor<double, n> forward(Tensor<T, n> &in) {
    std::array<size_t, n> one_shape = in.get_shape();
    one_shape[n - 1] = 1;
    Tensor<T, n> ones = Flint::constant_array<T, n>(1, one_shape);
    return Flint::concat(in, ones, n - 1).matmul(get_weight<0>());
  }
};
/** Randomly sets some values in the input to 0 with a probability of `p`.
 * Reduces over fitting. Degenerates to an identity function when `training` is
 * false. */
class Dropout : public UntrainableLayer {
  double p;

public:
  static constexpr FType transform_type(FType t) { return F_FLOAT64; }
  Dropout(double p) : p(p) {}
  template <typename T, unsigned int n>
  Tensor<double, n> forward(Tensor<T, n> &in) {
    if (!training) {
      if constexpr (std::is_same<T, double>()) {
        return in;
      } else {
        return in.template convert<double>();
      }
    }
    Tensor<double, n> r = Flint::random_array(in.get_shape());
    return (in * (r > p)) / (1.0 - p);
  }
};
struct Flatten : public UntrainableLayer {
  static constexpr unsigned int transform_dimensionality(unsigned int n) {
    return 2;
  }
  /** Flattens every feature axis into one single axis, does not touch the
   * batch-axis (the first) */
  template <typename T, unsigned int n>
  Tensor<T, 2> forward(const Tensor<T, n> &in) {
    if constexpr (n == 2)
      return in;
    else
      return forward(in.flattened(n - 1));
  }
};
#endif
