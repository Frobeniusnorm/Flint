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
#ifndef FLINT_ACTIVATIONS
#define FLINT_ACTIVATIONS
#include "layers.hpp"

/** SoftMax activation Layer. For multiclass classification. */
class SoftMax : public UntrainableLayer {
  int ax = -1;

public:
  /** Initializes the SoftMax function with an optional axis parameter that
   * describes the dimension of which the sum will be taken (may be negative in
   * which case it will index from back, i.e. -1 means the last axis, -2 the one
   * befor the last etc.). Calculates `exp(in) / sum(in, ax)` */
  SoftMax(int ax = -1) : ax(ax) {}

  template <typename T, unsigned int n> Tensor<T, n> forward(Tensor<T, n> &in) {
    unsigned int axis = ax < 0 ? n + ax : ax;
    // numerical stability
    Tensor<T, n> exp =
        (in - in.reduce_max(axis).expand(axis, in.get_shape()[axis])).exp();
    Tensor<T, n - 1> sum = exp.reduce_sum(axis);
    if (ax == 0 || n == 1)
      return exp / sum;
    else {
      return exp / (sum.expand(axis, in.get_shape()[axis]));
    }
  }
};
struct Relu : public UntrainableLayer {
  template <typename T, unsigned int n> Tensor<T, n> forward(Tensor<T, n> &in) {
    return in.max(0);
  }
};
#endif
