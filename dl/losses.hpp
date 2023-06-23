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
struct Loss {
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

class SoftMax : public Loss {
  int ax = -1;
public:
  template <typename T, unsigned int n>
  Tensor<T, n> forward(Tensor<T, n> &in) {
    // TODO this can be done way more accurate and efficient
    Tensor<T, n> exp = Flint::constant((T)2.71828, in.get_shape()).pow(in);
    return exp / exp.reduce_sum(ax < 0 ? in.get_shape()[n - ax] : in.get_shape()[ax]);
  }
};
