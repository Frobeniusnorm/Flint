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
#include <flint/flint.h>
#include <flint/flint.hpp>
#include <limits>
#include <optional>
#include <type_traits>
#include <unordered_set>
/**
 * Optimizer interface that defines a update method.
 * An optimizer is intended to be instantiated once per weight
 * and optimizes only double weights (since the gradient is also always given as
 * a double Tensor). It uses the C interface for easy extensibility. */
template <int n> struct Optimizer {
  virtual ~Optimizer() = default;
  virtual Tensor<double, n> update(Tensor<double, n> &weights,
                                   Tensor<double, n> &gradient) = 0;
};
template <typename T>
concept OptimizerFactory = requires(T fac) {
  {
    (fac.template generate_optimizer<2>())
  } -> std::convertible_to<Optimizer<2> *>;
};
/**
 * Implementation of the Adam algorithm (first-order gradient-based optimizer
 * for stochastic objective functions based on adaptive estimates of lower-order
 * moments)
 */
template <int n> struct Adam : public Optimizer<n> {
  double epsilon = std::numeric_limits<double>::epsilon();
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
  Adam(double learning_rate = 0.0015, double b1 = 0.9, double b2 = 0.999)
      : learning_rate(learning_rate), b1(b1), b2(b2) {}
  Tensor<double, n> update(Tensor<double, n> &weight, Tensor<double, n> &grad) {
    if (!init) {
      init = true;
      m = Flint::constant_array(0.0, weight.get_shape());
      v = Flint::constant_array(0.0, weight.get_shape());
    }
    grad.execute();
    m = m * b1 + grad * (1 - b1);
    v = v * b2 + grad * grad * (1 - b2);
    m.execute();
    v.execute();
    Tensor<double, n> mh = m / (1 - std::pow(b1, t));
    Tensor<double, n> vh = v / (1 - std::pow(b2, t));
    t += 1;
    return weight - (mh * learning_rate) / (vh.sqrt() + epsilon);
  }

private:
  Tensor<double, n> m;
  Tensor<double, n> v;
  bool init = false;
  unsigned long t = 1;
};
/**
 * Constructs Adam Optimizer with preset parameters.
 */
struct AdamFactory {
  double learning_rate, b1, b2;
  /** Initialisation parameters for the Adam algorithm with that influence the
   * optimization speed and accuracy.
   *  - `learning_rate`: (sometimes called `alpha`) the step size per
   *     optimization, i.e. the proportion weights are updated. Higher values (e.g.
   *     0.2) lead to a faster convergence, while lower values yield more accurate
   *     convergence.
   *  - `b1`: (sometimes called `beta1`) the exponential decay rate for the
   *     first moment estimates.
   *  - `b2`: (sometimes called `beta2`) the exponential decay rate for the
   *     second moment estimates.
   */
  AdamFactory(double learning_rate = 0.0015, double b1 = 0.9, double b2 = 0.999)
      : learning_rate(learning_rate), b1(b1), b2(b2) {}
  template <int n> Optimizer<n> *generate_optimizer() const {
    return new Adam<n>(learning_rate, b1, b2);
  }
};

#endif
