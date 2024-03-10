/* Copyright 2023 David Schwarzbeck
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */
#ifndef FLINT_NORMALIZATION
#define FLINT_NORMALIZATION
#include "../layers.hpp"
#include <limits>

/** Randomly sets some values in the input to 0 with a probability of `p`.
 * Reduces over fitting. Degenerates to an identity function when `training` is
 * false. */
class Dropout : public UntrainableLayer {
		double p;

	public:
		Dropout() : p(0.1) {}
		Dropout(double p) : p(p) {}
		template <typename T, unsigned int n>
		Tensor<T, n> forward(Tensor<T, n> &in) {
			if (!training) {
				return in;
			}
			// s.t. the memory may be reused we remove the handle of in
			in.get_graph_node()->reference_counter--;
			auto result = in.dropout(p);
			// avoid in freeing memory on destruction
			in.set_graph_node(nullptr);
			return result;
		}
		std::string name() override { return "Dropout"; }
		std::string description() override {
			return "p = " + std::to_string(p);
		}
};
template <unsigned int n, typename F = float>
struct LayerNorm : public Layer<F, n, n> {
		LayerNorm(std::array<size_t, n> normalized_shape)
			: Layer<F, n, n>(Flint::constant_array(1.0f, normalized_shape),
							 Flint::constant_array(0.0f, normalized_shape)) {}
		template <typename T, unsigned int k>
		Tensor<T, k> forward(Tensor<T, k> &in) {
			// flatten last dimensions for reductions
			using namespace std;
			array<size_t, k - n + 1> flat_shape;
			for (int i = 0; i < k - n; i++)
				flat_shape[i] = in.get_shape()[i];
			flat_shape[k - n] = 1;
			for (int i = 0; i < n; i++)
				flat_shape[k - n] *= in.get_shape()[k - n + i];
			Tensor<T, k - n + 1> flat = in.reshape_array(flat_shape);
			Tensor<T, k - n> mean =
				(flat / (T)flat.get_shape()[k - n]).reduce_sum();
			Tensor<T, k - n> var =
				((flat - mean).pow(2) / (T)flat.get_shape()[k - n])
					.reduce_sum();
			return ((in - mean) / (var + numeric_limits<F>::epsilon()).sqrt()) *
					   Layer<F, n, n>::template get_weight<0>() +
				   Layer<F, n, n>::template get_weight<1>();
		}
};
#endif
