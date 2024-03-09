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
#ifndef FLINT_CONNECTED
#define FLINT_CONNECTED
#include "../layers.hpp"
/**
 * Layer for fully connected neuronal network layer.
 * A connected layer has a 2 dimensional matrix and a bias as parameters.
 * The matrix is multiplied (with matrix multiplication) with the last two
 * dimensions of the input tensor. The bias is added on the result (in practice
 * this happens in one matrix multiplication, the input tensor is padded with a
 * 1 in its last dimension and the bias is the last row of the matrix).
 */
template <typename F = float> struct Connected : public Layer<F, 2> {
		/**
		 * Creates the layer and initializes the weights.
		 * - `units_in` size of the last dimension of the input tensors (will be
		 * the size of the dimension before the last dimension of the weights).
		 * - `units_out` size of the last dimension the result tensor is
		 * supposed to have (will be the size of the last dimension of the
		 * weights).
		 * - `init_weights` a weight initializer (has to fulfill the
		 * `Initializer` concept, close to Gauss-distributed random values yield
		 * good results).
		 * - `init_bias` a bias initializer (has to fulfill the `Initializer`
		 * concept, small values yield good results, can be constant for bias).
		 */
		template <Initializer InitWeights, Initializer InitBias>
		Connected(size_t units_in, size_t units_out, InitWeights init_weights,
				  InitBias init_bias)
			: Layer<F, 2>(
				  Flint::concat(init_weights.template initialize<F>(
									std::array<size_t, 2>{units_in, units_out}),
								init_bias.template initialize<F>(
									std::array<size_t, 2>{1, units_out}),
								0)) {}
		/**
		 * Creates the layer and initializes the weights.
		 * - `units_in` size of the last dimension of the input tensors (will be
		 * the size of the dimension before the last dimension of the weights).
		 * - `units_out` size of the last dimension the result tensor is
		 * supposed to have (will be the size of the last dimension of the
		 * weights).
		 *
		 * The weights are initialized with glorot uniform random values and
		 * the bias with 0s.
		 */
		Connected(size_t units_in, size_t units_out)
			: Layer<F, 2>(Flint::concat(
				  GlorotUniform().template initialize<F>(
					  std::array<size_t, 2>{units_in, units_out}),
				  ConstantInitializer().template initialize<F>(
					  std::array<size_t, 2>{1, units_out}),
				  0)) {}
		template <typename T, unsigned int n>
		Tensor<
			LayerHelper::FlintTypeToCpp<Layer<F, n, 1>::transform_type(to_flint_type<T>())>,
			n>
		forward(Tensor<T, n> &in) {
			std::array<size_t, n> one_shape = in.get_shape();
			one_shape[n - 1] = 1;
			// allow optimization of in
			in.get_graph_node()->reference_counter--;
			// create one column for bias
			Tensor<T, n> ones = Flint::constant_array<T, n>(1, one_shape);
			auto a = Flint::concat(in, ones, n - 1);
			// prevent destructor to remove memory
			in.set_graph_node(nullptr);
			return a.matmul(Layer<F, 2>::template get_weight<0>());
		}
		std::string name() override { return "Connected"; }
		std::string description() override {
			return std::to_string(Layer<F, 2>::template get_weight<0>().get_shape()[0]) + " * " +
				   std::to_string(Layer<F, 2>::template get_weight<0>().get_shape()[1]);
		}
};
#endif
