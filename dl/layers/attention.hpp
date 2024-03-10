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
#ifndef FLINT_ATTENTION
#define FLINT_ATTENTION
#include "dl/activations.hpp"
#include "dl/layers.hpp"
#include "dl/layers/connected.hpp"
struct PositionalEncoding : public UntrainableLayer {

		PositionalEncoding(size_t d_model, size_t seq_space) {
			Tensor<float, 2> pos = Flint::arange(0, seq_space, 1)
									   .repeat(0, (d_model / 2) - 1)
									   .convert<float>();
			Tensor<float, 1> div =
				Flint::arange(0, (d_model - 1) / 2 + 1) * 2.0f;
			Tensor<float, 2> even = (pos * div).sin();
			Tensor<float, 2> odd = (pos * div).cos();
			encoding = even.extend({seq_space, d_model}, {0, 0}, {1, 2}) +
					   odd.extend({seq_space, d_model}, {0, 1}, {1, 2});
			encoding.execute();
		}
		template <typename T, unsigned int n>
		Tensor<LayerHelper::FlintTypeToCpp<
				   UntrainableLayer::transform_type(to_flint_type<T>())>,
			   n>
		forward(Tensor<T, n> &in) {
			return in + encoding.slice(TensorRange::MAX_SCOPE,
									   TensorRange(0, in.get_shape()[1]));
		}

	private:
		Tensor<float, 2> encoding;
};
template <typename F = float>
using MultiheadAttentionComposer =
	ComposerLayer<Connected<F>, Connected<F>, Connected<F>, Connected<F>>;
template <typename F = float>
struct MultiheadAttention : MultiheadAttentionComposer<F> {
		template <Initializer InitWeights, Initializer InitBias>
		MultiheadAttention(size_t num_heads, size_t d_model,
						   InitWeights init_weights, InitBias init_bias)
			: MultiheadAttentionComposer<F>(
				  Connected<F>(d_model, d_model, init_weights, init_bias),
				  Connected<F>(d_model, d_model, init_weights, init_bias),
				  Connected<F>(d_model, d_model, init_weights, init_bias),
				  Connected<F>(d_model, d_model, init_weights, init_bias)),
			  num_heads(num_heads), d_model(d_model) {
			if (d_model % num_heads != 0)
				flogging(F_ERROR, "Error in Multihead Attention: d_model must "
								  "be a multiple of num_heads");
		}
		static constexpr FType transform_type(FType t) {
			return higher_type_constexpr(t, to_flint_type<F>());
		}
		std::string name() { return "Multihead Attention"; }
		std::string description() { return ""; }
		template <typename T, unsigned int n>
		Tensor<LayerHelper::FlintTypeToCpp<transform_type(to_flint_type<T>())>,
			   n>
		forward(Tensor<T, n> &in) {
			if (n != 4)
				flogging(F_WARNING,
						 "Multihead Attention with a dimensionality of the "
						 "input tensor != 4 is undefined! This is probably not "
						 "what you want to do.");
			if (in.get_shape()[0] != 3)
				flogging(F_ERROR,
						 "Multihead Attention expects the first dimension to "
						 "be of size 3, holding query, key and value");
			// destruct input
			Tensor<F, n - 1> query = in.slice(TensorRange(0, 1));
			Tensor<F, n - 1> key = in.slice(TensorRange(1, 2));
			Tensor<F, n - 1> value = in.slice(TensorRange(2, 3));
			// pass through linear layers
			Tensor<F, 4> Q =
				split_heads(std::get<0>(MultiheadAttentionComposer<F>::layers)
								.forward(query));
			Tensor<F, 4> K =
				split_heads(std::get<1>(MultiheadAttentionComposer<F>::layers)
								.forward(key));
			Tensor<F, 4> V =
				split_heads(std::get<2>(MultiheadAttentionComposer<F>::layers)
								.forward(value));
			// compute attention
			Tensor<F, 4> score = Q.matmul(K.transpose(0, 1, 3, 2)) /
								 sqrt((int)(d_model / num_heads));
			score = softmax.forward(score);
			score = score.matmul(V);
			// combine heads
			size_t batch_size = in.get_shape()[0];
			size_t seq_length = in.get_shape()[2];
			Tensor<F, 3> combined =
				score.transpose(0, 2, 1, 3)
					.reshape(batch_size, seq_length, d_model);
			return std::get<3>(MultiheadAttentionComposer<F>::layers)
				.forward(combined);
		}

	private:
		template <unsigned int n> Tensor<F, 4> split_heads(Tensor<F, n> &in) {
			size_t batch_size = in.get_shape()[0];
			size_t seq_length = in.get_shape()[1];
			return in
				.reshape(batch_size, seq_length, num_heads,
						 this->d_model / num_heads)
				.transpose(0, 2, 1, 3);
		}
		size_t num_heads, d_model;
		SoftMax softmax; // so i dont have to reimplement it
};
#endif
