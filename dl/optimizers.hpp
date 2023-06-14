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
#include <flint/flint.h>
#include <optional>
/**
 * Optimizer interface that defines a update method.
 * An optimizer is intended to be instantiated once per weight
 * and optimizes only double weights (since the gradient is also always given as
 * a double Tensor). It uses the C interface for easy extensibility. */
struct Optimizer {
  virtual ~Optimizer() = default;
  virtual FGraphNode *update(FGraphNode *weights, FGraphNode *gradient) = 0;
};
struct OptimizerFactory {
  virtual Optimizer *generate_optimizer() const = 0;
};
/**
 * Implementation of the Adam algorithm (first-order gradient-based optimizer
 * for stochastic objective functions based on adaptive estimates of lower-order
 * moments)
 */
struct Adam : public Optimizer {
  double epsilon = 1e-08;
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
  Adam(const Adam &other) {
    if (other.m)
      m = fCopyGraph(other.m);
    if (other.v)
      v = fCopyGraph(other.v);
    learning_rate = other.learning_rate;
    b1 = other.b1;
    b2 = other.b2;
    t = other.t;
  }
  Adam(Adam &&other) {
    if (other.m)
      m = other.m;
    if (other.v)
      v = other.v;
    other.m = nullptr;
    other.v = nullptr;
    learning_rate = other.learning_rate;
    b1 = other.b1;
    b2 = other.b2;
    t = other.t;
  }
  Adam &operator=(const Adam &other) {
    if (m && v) {
      m->reference_counter--;
      v->reference_counter--;
      fFreeGraph(m);
      fFreeGraph(v);
    }
    if (other.m)
      m = fCopyGraph(other.m);
    if (other.v)
      v = fCopyGraph(other.v);
    learning_rate = other.learning_rate;
    b1 = other.b1;
    b2 = other.b2;
    t = other.t;
    return *this;
  }
  Adam &operator=(Adam &&other) {
    if (m && v) {
      m->reference_counter--;
      v->reference_counter--;
      fFreeGraph(m);
      fFreeGraph(v);
    }
    if (other.m)
      m = other.m;
    if (other.v)
      v = other.v;
    other.m = nullptr;
    other.v = nullptr;
    learning_rate = other.learning_rate;
    b1 = other.b1;
    b2 = other.b2;
    t = other.t;
    return *this;
  }

  ~Adam() {
    if (m && v) {
      m->reference_counter--;
      v->reference_counter--;
      fFreeGraph(m);
      fFreeGraph(v);
    }
  }
  FGraphNode *update(FGraphNode *weights, FGraphNode *gradient) {
    if (!m) {
      // initialize with weight shape
      m = fconstant_d(0, weights->operation->shape,
                      weights->operation->dimensions);
      v = fconstant_d(0, weights->operation->shape,
                      weights->operation->dimensions);
      m->reference_counter++;
      v->reference_counter++;
    }
    m = fadd(fmul_cd(m, b1), fmul_cd(gradient, (1 - b1)));
    v = fadd(fmul_cd(v, b2), fmul(fmul_cd(gradient, (1 - b2)), gradient));
    FGraphNode *mh = fdiv_cd(m, (1 - std::pow(b1, t)));
    FGraphNode *vh = fdiv_cd(v, (1 - std::pow(b2, t)));
    t++;
    return fsub(weights, fdiv(fmul_cd(mh, learning_rate),
                              fadd_cd(fsqrt_g(vh), epsilon)));
  }

  template <int n>
  Tensor<double, n> update(Tensor<double, n> &weights,
                           Tensor<double, n> &gradient) {
    return Tensor<double, n>(
        update(weights.get_graph_node(), gradient.get_graph_node()),
        weights.get_shape());
  }

private:
  FGraphNode *m = nullptr; // 1st moment
  FGraphNode *v = nullptr; // 2nd moment
  unsigned long t = 1;
};
struct AdamFactory : public OptimizerFactory {
  double epsilon = 1e-07;
  double learning_rate, b1, b2;
  AdamFactory(double learning_rate = 0.0015, double b1 = 0.9, double b2 = 0.999)
      : learning_rate(learning_rate), b1(b1), b2(b2) {}
  Optimizer *generate_optimizer() const { return new Adam(learning_rate, b1, b2); }
};
#endif
