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
#include "losses.hpp"
#include "optimizers.hpp"
#include <flint/flint.h>
#include <flint/flint_helper.hpp>
#include <iomanip>
#include <math.h>
#include <memory>
#include <tuple>
#include <vector>

template <FType in> constexpr FType get_output_type() { return in; }
template <FType in, GenericLayer K> constexpr FType get_output_type() {
  return K::transform_type(in);
}
template <FType in, GenericLayer K1, GenericLayer K2, GenericLayer... F>
constexpr FType get_output_type() {
  constexpr FType out = K2::transform_type(K1::transform_type(in));
  return get_output_type<out, F...>();
}
template <unsigned int in> constexpr unsigned int get_output_dim() {
  return in;
}
template <unsigned int in, GenericLayer K>
constexpr unsigned int get_output_dim() {
  return K::transform_dimensionality(in);
}
template <unsigned int in, GenericLayer K1, GenericLayer K2, GenericLayer... F>
constexpr unsigned int get_output_dim() {
  constexpr unsigned int out =
      K2::transform_dimensionality(K1::transform_dimensionality(in));
  return get_output_dim<out, F...>();
}

template <GenericLayer... T> struct SequentialModel {
  std::tuple<T...> layers;
  SequentialModel(T... layers) : layers(std::move(layers)...) {}

  template<OptimizerFactory Fac>
  void generate_optimizer(Fac fac) { gen_opt<0>(fac); }

  template <typename K, unsigned int n>
  Tensor<LayerHelper::FlintTypeToCpp<get_output_type<toFlintType<K>(), T...>()>,
         get_output_dim<n, T...>()>
  forward(Tensor<K, n> &in) {
    return forward_helper<
        0,
        LayerHelper::FlintTypeToCpp<get_output_type<toFlintType<K>(), T...>()>,
        get_output_dim<n, T...>()>(in);
  }
  template <typename K, unsigned int n>
  void optimize(const Tensor<K, n> &error) {
    backward<0>(error);
  }
  // TODO train with Datagenerators
  template <typename T1, unsigned int n1, typename T2, unsigned int n2,
            GenericLoss L>
  void train(Tensor<T1, n1> &X, Tensor<T2, n2> &Y, L loss, int epochs = 1,
             int batch_size = 32) {
    set_training<0>(true);
    const size_t batches = X.get_shape()[0];
    if (Y.get_shape()[0] != batches)
      flogging(F_ERROR,
               "Input and Target Datas batch size does not correspond!");
    std::cout << "\r\e[Kbatch error: ... \e[1;30m" ;
    for (int k = 0; k < 15; k++)
      std::cout << "―";
    std::cout << "\033[0m" << std::flush;
    for (int i = 0; i < epochs; i++) {
      // TODO shuffle each iteration
      size_t number_batches = batches / batch_size + 1;
      double total_error = 0;
      for (size_t b = 0; b < number_batches; b++) {
        size_t slice_to = (b + 1) * batch_size;
        if (slice_to > batches)
          slice_to = batches;
        if (b * batch_size == slice_to)
          break;
        // run batch and calculate error
        auto input = X.slice(TensorRange(b * batch_size, slice_to));
        auto expected = Y.slice(TensorRange(b * batch_size, slice_to));
        input.execute();
        expected.execute();
        fStartGradientContext();
        auto output = forward(input);
        auto error = loss.calculate_error(output, expected);
        fStopGradientContext();
        // optimize weights 
        backward<0>(error); // TODO remove
        if (false) { // TODO does not work yet!
          std::vector<std::vector<FGraphNode*>> vars;
          collect_weights<0>(vars);
          std::vector<FGraphNode*> flat_vars;
          for (unsigned int i = 0; i < vars.size(); i++)
            flat_vars.insert(flat_vars.end(), vars[i].begin(), vars[i].end());
          std::vector<FGraphNode*> grads(flat_vars.size());
          // fCalculateGradients(error.get_graph_node(), flat_vars.data(), flat_vars.size(), grads.data());
          // REPLACEMENT:
          for (unsigned int i = 0; i < flat_vars.size(); i++)
            grads[i] = fOptimizeMemory(fExecuteGraph(fCalculateGradient(error.get_graph_node(), flat_vars[i])));
          // END
          std::vector<std::vector<FGraphNode*>> plgrads(vars.size());
          int index = 0;
          for (unsigned int i = 0; i < vars.size(); i++) {
            plgrads[i] = std::vector<FGraphNode*>(vars[i].size());
            for (unsigned int j = 0; j < vars[i].size(); j++)
              plgrads[i][j] = grads[index++];
          }
        }
        double local_error = (double)(error.reduce_sum()[0]);
        total_error += local_error / number_batches;
        // print metrics
        std::cout << "\r\e[Kbatch error: " << std::setprecision(3)
                  << local_error << " \e[1;96m";
        for (int k = 0; k < 15; k++) {
          if ((k + 1) / 15.0 <= (b + 1.0) / number_batches)
            std::cout << "―";
          else {
            std::cout << "\e[1;30m";
            for (int l = k; l < 15; l++)
              std::cout << "―";
            break;
          }
        }
        std::cout << "\033[0m" << std::flush;
      }
      std::cout << "\r\e";
      flogging(F_INFO, "Mean loss for epoch #" + std::to_string(i + 1) + ": " +
                           std::to_string(total_error));
    }
    set_training<0>(false);
  }

private:
  template <int n, typename K, unsigned int k>
  void backward(const Tensor<K, k> &error) {
    if constexpr (n < sizeof...(T)) {
      std::get<n>(layers).optimize_weights(error);
      backward<n + 1>(error);
    }
  }
  template <int n>
  void backward(const std::vector<std::vector<FGraphNode*>> grads) {
    if constexpr (n < sizeof...(T)) {
      std::get<n>(layers).optimize_weights(grads[n]);
      backward<n + 1>(grads);
    }
  }
  template <int n, OptimizerFactory Fac>
  void gen_opt(Fac fac) {
    if constexpr (n < sizeof...(T)) {
      std::get<n>(layers).generate_optimizer(fac);
      gen_opt<n + 1>(fac);
    }
  }
  template <int n> void set_training(bool b) {
    if constexpr (n < sizeof...(T)) {
      std::get<n>(layers).training = b;
      set_training<n + 1>(b);
    }
  }
  template <int n> void collect_weights(std::vector<std::vector<FGraphNode*>>& vars) {
    if constexpr (n < sizeof...(T)) {
      vars.push_back(std::get<n>(layers).collect_weights());
      collect_weights<n + 1>(vars);
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
