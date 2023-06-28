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
#include "layers.hpp"
#include <concepts>
#include <flint/flint.hpp>
#include <limits>
/**
 * Defines the general concept of a Loss function.
 * It receives two tensors: the actual output and the expected one.
 * It then calculates the loss as a double Tensor (since the weights are always
 * double Tensors as well).
 */
template <typename T>
concept GenericLoss =
    requires(T a, Tensor<float, 2> &t1, Tensor<int, 2> &t2,
             Tensor<double, 2> &t3, Tensor<long, 2> &t4) {
      {
        a.calculate_error(t1, t1)
        } -> std::convertible_to<
            Tensor<LayerHelper::FlintTypeToCpp<T::transform_type(F_FLOAT32)>,
                   T::transform_dimensionality(2)>>;
      { a.calculate_error(t2, t2) } -> std::convertible_to<
            Tensor<LayerHelper::FlintTypeToCpp<T::transform_type(F_FLOAT32)>,
                   T::transform_dimensionality(2)>>;
      { a.calculate_error(t3, t3) } -> std::convertible_to<
            Tensor<LayerHelper::FlintTypeToCpp<T::transform_type(F_FLOAT32)>,
                   T::transform_dimensionality(2)>>;
      { a.calculate_error(t4, t4) } -> std::convertible_to<
            Tensor<LayerHelper::FlintTypeToCpp<T::transform_type(F_FLOAT32)>,
                   T::transform_dimensionality(2)>>;
    };

/** Calculates the Categorical Cross Entropy Loss with full summation. It is
 * advised to apply a softmax as the last activation layer in the calculation of
 * `in`.
 *
 * Calculates: `sum(-expected * log(in))`
 * */
struct CrossEntropyLoss {
  static constexpr FType transform_type(FType t) { return F_FLOAT64; }
  static constexpr unsigned int transform_dimensionality(int n) { return n - 1 < 1 ? 1 : n - 1; }
  template <typename T, unsigned int n>
  Tensor<double, transform_dimensionality(n)> calculate_error(Tensor<T, n> &in,
                                        Tensor<T, n> &expected) {
    // TODO sparse categorical cross entropy loss
    size_t total_size = 1;
    for (int i = 0; i < n - 1; i++)
      total_size *= in.get_shape()[i];
    return (-(expected * (in + 1e-9).log()).reduce_sum() / (double)total_size);
  }
};
