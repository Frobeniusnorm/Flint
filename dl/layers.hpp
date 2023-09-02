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
#include <chrono>
namespace LayerHelper {
/**
 * FOR INTERNAL USE ONLY
 * builds an compile-time linked list of Tensor pointer
 */
template <unsigned int index, int... w> class WeightRef {};

template <unsigned int index, int n> struct WeightRef<index, n> {
  Tensor<double, n> weight;
  std::unique_ptr<Optimizer<n>> optimizer = nullptr;
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
  template <OptimizerFactory Fac> void gen_optimizer(const Fac fac) {
    optimizer =
        std::unique_ptr<Optimizer<n>>(fac.template generate_optimizer<n>());
  }
  template <typename T, unsigned int k>
  void optimize(const Tensor<T, k> &error) {
    if (optimizer) {
      Tensor<double, n> gw =
          error.gradient(weight); // this line - somehow - is the problem
      Tensor<double, n> nw = optimizer->update(weight, gw);
      nw.execute();
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
  void collect_weights(std::vector<FGraphNode *> &nodes) {
    nodes[index] = weight.get_graph_node();
  }
  void update_weights(std::vector<FGraphNode *> &grads) {
    if (optimizer) {
#ifdef FLINT_DL_PROFILE
      auto start = std::chrono::high_resolution_clock::now();
#endif
      Tensor<double, n> gw(grads[index], weight.get_shape());
#ifdef FLINT_DL_PROFILE
      std::chrono::duration<double, std::milli> elapsed =
        std::chrono::high_resolution_clock::now() - start;
      flogging(F_INFO, "weights update took " + std::to_string(elapsed.count()) + "ms");
#endif
      Tensor<double, n> nw = optimizer->update(weight, gw);
      nw.execute();
      weight = std::move(nw);
      weight.watch();
    } else {
      flogging(F_WARNING, "No Optimizer for weight!");
    }
  }
};

template <unsigned int index, int n, int... wn>
struct WeightRef<index, n, wn...> {
  Tensor<double, n> weight;
  std::unique_ptr<Optimizer<n>> optimizer = nullptr;
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
  template <OptimizerFactory Fac> void gen_optimizer(const Fac fac) {
    optimizer =
        std::unique_ptr<Optimizer<n>>(fac.template generate_optimizer<n>());
    others.gen_optimizer(fac);
  }
  template <typename T, unsigned int k>
  void optimize(const Tensor<T, k> &error) {
    if (optimizer) {
      Tensor<double, n> gw = error.gradient(weight);
      Tensor<double, n> nw = optimizer->update(weight, gw);
      nw.execute();
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
  void collect_weights(std::vector<FGraphNode *> &nodes) {
    nodes[index] = weight.get_graph_node();
    others.collect_weights(nodes);
  }
  void update_weights(std::vector<FGraphNode *> &grads) {
    if (optimizer) {
      Tensor<double, n> gw(grads[index], weight.get_shape());
      Tensor<double, n> nw = optimizer->update(weight, gw);
      weight = std::move(nw);
      weight.watch();
    } else {
      flogging(F_WARNING, "No Optimizer for weight!");
    }
    others.update_weights(grads);
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
             Tensor<double, 2> &t3, Tensor<long, 2> &t4, AdamFactory fac,
             std::vector<FGraphNode *> grads) {
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
      { a.collect_weights() } -> std::convertible_to<std::vector<FGraphNode *>>;
      a.optimize_weights(grads);
      a.generate_optimizer(fac);
      a.training = true;
      { T::transform_dimensionality(5) } -> std::convertible_to<unsigned int>;
      { T::transform_type(F_INT32) } -> std::convertible_to<FType>;
      { a.name() } -> std::convertible_to<std::string>;
      { a.summary() } -> std::convertible_to<std::string>;
    };
/** Implements blank methods for every method of GenericLayer that is not needed
 * for a Layer that is not trainable */
struct UntrainableLayer {
  bool training = false;
  // to fulfill generic layer
  template <OptimizerFactory Fac> void generate_optimizer(Fac factory) {}
  // to fulfill generic layer
  template <typename T, unsigned int dim>
  void optimize_weights(const Tensor<T, dim> &error) {}
  void optimize_weights(std::vector<FGraphNode *> grads) {}
  std::vector<FGraphNode *> collect_weights() { return {}; }
  static constexpr FType transform_type(FType t) { return t; }
  static constexpr unsigned int transform_dimensionality(unsigned int n) {
    return n;
  }
  virtual std::string name() {
    return "unnamed";
  }
  virtual std::string summary() {
    return name() + " layer";
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
  template <OptimizerFactory Fac> void generate_optimizer(Fac factory) {
    weight_refs.gen_optimizer(factory);
  }
  template <typename T, unsigned int dim>
  void optimize_weights(const Tensor<T, dim> &error) {
    weight_refs.optimize(error);
  }
  std::vector<FGraphNode *> collect_weights() {
    std::vector<FGraphNode *> nodes(sizeof...(wn));
    weight_refs.collect_weights(nodes);
    return nodes;
  }
  void optimize_weights(std::vector<FGraphNode *> grads) {
    weight_refs.update_weights(grads);
  }
  virtual std::string name() {
    return "unnamed";
  }
  virtual std::string summary() {
    return name() + " layer with " + std::to_string(sizeof...(wn)) +
           " weight tensors";
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
  std::string name() override{
    return "Connected";
  }
  std::string summary() override {
    return name() + ": " + std::to_string(get_weight<0>().get_shape()[0]) +
           " * " + std::to_string(get_weight<0>().get_shape()[1]);
  }
};
enum PaddingMode { PADDING_END, NO_PADDING };
template <int n> class Convolution : public Layer<n> {
  constexpr std::array<size_t, n> weight_shape(unsigned int filters,
                                               unsigned int kernel_size,
                                               size_t units_in) {
    std::array<size_t, n> res;
    res[0] = filters;
    for (int i = 1; i < n - 1; i++) {
      res[i] = kernel_size;
    }
    res[n - 1] = units_in;
    return res;
  }
  std::array<unsigned int, n - 1> act_stride;
  unsigned int kernel_size;
  void initialize_precalc(std::array<unsigned int, n - 2> stride) {
    act_stride[0] = 1;
    for (int i = 0; i < n - 2; i++)
      act_stride[i + 1] = stride[i];
  }

public:
  PaddingMode padding_mode;
  // weights have shape: filter, kernel size, units
  template <Initializer InitWeights>
  Convolution(size_t units_in, unsigned int filters, unsigned int kernel_size,
              InitWeights init, std::array<unsigned int, n - 2> stride,
              PaddingMode padding_mode = NO_PADDING)
      : Layer<n>(init.template initialize<double>(
            weight_shape(filters, kernel_size, units_in))),
        padding_mode(padding_mode), kernel_size(kernel_size) {
    initialize_precalc(stride);
  }

  Convolution(size_t units_in, unsigned int filters, unsigned int kernel_size,
              std::array<unsigned int, n - 2> stride,
              PaddingMode padding_mode = NO_PADDING)
      : Layer<n>(GlorotUniform().template initialize<double>(
            weight_shape(filters, kernel_size, units_in))),
        padding_mode(padding_mode), kernel_size(kernel_size) {

    initialize_precalc(stride);
  }
  std::string name() override{
    return "Convolution";
  }
  std::string summary() override {
    const unsigned int filters =
        Layer<n>::template get_weight<0>().get_shape()[0];
    const unsigned int units_in =
        Layer<n>::template get_weight<0>().get_shape()[n - 1];
    const unsigned int kernel_size =
        Layer<n>::template get_weight<0>().get_shape()[1];
    return name() + ": input channels: " + std::to_string(units_in) +
           " filters: " + std::to_string(filters) +
           ", kernel size: " + std::to_string(kernel_size);
  }
  template <typename T, unsigned int k>
  Tensor<double, k> forward(Tensor<T, k> &in) {
    const unsigned int filters =
        Layer<n>::template get_weight<0>().get_shape()[0];
    // actual convolve
    Tensor<double, k> res;
    for (unsigned int i = 0; i < filters; i++) {
      // in has shape [batch, dim1, ..., units_in]
      // has shape [1, kernel_size, ..., units_in]
      Tensor<double, n> filter =
          Layer<n>::template get_weight<0>().slice(TensorRange(i, i + 1));
      Tensor<double, n - 1> filter_res = in.convolve_array(filter, act_stride);
      std::array<size_t, n> new_shape;
      for (int i = 0; i < n - 1; i++)
        new_shape[i] = filter_res.get_shape()[i];
      new_shape[n - 1] = 1;
      Tensor<double, k> local_res = filter_res.reshape_array(new_shape);
      local_res.execute();
      res = i == 0 ? local_res : Flint::concat(res, local_res, n - 1);
    }
    return res;
  }
};
typedef Convolution<4> Conv2D; // batch-size dim1 dim2 channels -> batch-size
                               // new_dim1 new_dim2 filters

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
    // Tensor<double, n> r = Flint::constant_array(1.0, in.get_shape());
    Tensor<double, n> o = (in * (r > p)) / (1.0 - p);
    return o;
  }
  std::string name() override{
    return "Dropout";
  }
  std::string summary() override {
    return name() + " (p = " + std::to_string(p) + ")";
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
      return Tensor<T, 2>(in);
    else
      return forward(in.flattened(n - 1));
  }
  std::string name() override {
    return "Flatten";
  }
};
#endif
