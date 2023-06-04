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
#ifndef FLINT_OPTIMIZERS
#define FLINT_OPTIMIZERS
#include <cmath>
#include <flint/flint.hpp>
#include <optional>
/**
 * Optimizer interface that defines a update method.
 * An optimizer is intended to be instantiated once per weight
 * and optimizes only double weights (since the gradient is also always given as
 * a double Tensor). `n` is the number of dimensions of the weight. */
template <int n> struct Optimizer {
  virtual Tensor<double, n> update(Tensor<double, n> weights,
                                   Tensor<double, n> gradient) = 0;
};
/**
 * Implementation of the Adam algorithm (first-order gradient-based optimizer
 * for stochastic objective functions based on adaptive estimates of lower-order
 * moments)
 */
template <int n> struct Adam : public Optimizer<n> {
  double epsilon = 1e-07;
  double learning_rate, b1, b2;
  /**
   * Initializes the Adam algorithm with some parameters that influence the
   * optimization speed and accuracy.
   *  - `learning_rate`: (sometimes called `alpha`) the step size per
   * optimization, i.e. the proportion weights are updated. Higher values (e.g.
   * 0.2) lead to a faster convergence, while lower values yield more accurate
   * convergence.
   *  - `b1`: (sometimes called `beta1`) the exponential decay rate for the
   *     first moment estimates.
   *  - `b2`: (sometimes called `beta2`) the exponential decay rate for the
   *     second moment estimates.
   *
   * You can tune the individual members later on too.
   */
  Adam(double learning_rate = 0.05, double b1 = 0.9, double b2 = 0.999)
      : learning_rate(learning_rate), b1(b1), b2(b2) {}
  Tensor<double, n> update(Tensor<double, n> weights,
                           Tensor<double, n> gradient) {
    if (!m.has_value()) {
      // initialize with weight shape
      m = Tensor<double, n>::constant(0, weights.get_shape());
      v = Tensor<double, n>::constant(0, weights.get_shape());
    }
    m = m.value() * b1 + gradient * (1 - b1);
    v = v.value() * b2 + gradient * (1 - b2) * gradient;
    Tensor<double, n> mh = m.value() / (1 - std::pow(b1, t));
    Tensor<double, n> vh = v.value() / (1 - std::pow(b2, t));
    t++;
    return weights - mh * learning_rate  / (vh.sqrt() + epsilon);
  }

private:
  std::optional<Tensor<double, n>> m; // 1st moment
  std::optional<Tensor<double, n>> v; // 2nd moment
  unsigned long t = 1;
};

#endif
