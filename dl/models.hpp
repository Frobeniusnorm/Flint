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
#include "optimizers.hpp"
#include "layers.hpp"
#include <memory>
#include <tuple>
#include <vector>

template <typename... T> struct SequentialModel {
  std::tuple<T...> layers;
  SequentialModel() {
    static_assert((std::is_base_of_v<GenericLayer, T> && ...),
                  "SequentialModel only allows Layer that are derived from "
                  "GenericLayer!");
  }
  SequentialModel(T... layers) : layers(std::move(layers)...) {
    static_assert((std::is_base_of_v<GenericLayer, T> && ...),
                  "SequentialModel only allows Layer that are derived from "
                  "GenericLayer!");
  }
  void generate_optimizer(OptimizerFactory* fac) {
    gen_opt<0>(fac);
  }
private:
  template<int n>
  void gen_opt(OptimizerFactory* fac) {
    if constexpr (n < sizeof...(T)) {
      std::get<n>(layers).generate_optimizer(fac);
      gen_opt<n + 1>(fac);
    }
  }
};
#endif
