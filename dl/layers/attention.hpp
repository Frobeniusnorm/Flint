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
#include "dl/layers/normalization.hpp"
#include <utility>
//// positional encoding
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
//// helper layer
template <typename F = float>
using PWFFComposer = ComposerLayer<Connected<F>, Connected<F>>;

template <typename F = float>
struct PositionWiseFeedForward : public PWFFComposer<F> {
		Relu relu;
		template <Initializer InitWeights, Initializer InitBias>
		PositionWiseFeedForward(size_t d_model, size_t d_ff,
								InitWeights init_weights, InitBias init_bias)
			: PWFFComposer<F>(
				  Connected<F>(d_model, d_ff, init_weights, init_bias),
				  Connected<F>(d_ff, d_model, init_weights, init_bias)) {}
		PositionWiseFeedForward(size_t d_model, size_t d_ff)
			: PWFFComposer<F>(Connected<F>(d_model, d_ff),
							  Connected<F>(d_ff, d_model)) {}
		std::string name() { return "Position-wise Feed Forward"; }
		std::string description() { return ""; }
		template <typename T, unsigned int n>
		Tensor<LayerHelper::FlintTypeToCpp<transform_type(to_flint_type<T>())>,
			   n>
		forward(Tensor<T, n> &in) {
			auto x = std::get<0>(PWFFComposer<F>::layers).forward(in);
			x = std::get<1>(PWFFComposer<F>::layers).forward(x);
			return relu.forward(x);
		}
};
//// Attention
template <typename F = float>
using MultiheadAttentionComposer =
	ComposerLayer<Connected<F>, Connected<F>, Connected<F>, Connected<F>>;
// TODO: layer concept of Module that has no forward type check but all other
// advantages
// TODO: masks
template <typename F = float>
struct MultiheadAttention : public MultiheadAttentionComposer<F> {
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
		MultiheadAttention(size_t num_heads, size_t d_model)
			: MultiheadAttentionComposer<F>(Connected<F>(d_model, d_model),
											Connected<F>(d_model, d_model),
											Connected<F>(d_model, d_model),
											Connected<F>(d_model, d_model)),
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
		forward(Tensor<T, n> &query, Tensor<T, n> &key, Tensor<T, n> &value) {
			if (n != 3)
				flogging(F_WARNING,
						 "Multihead Attention with a dimensionality of the "
						 "input tensor != 4 is undefined! This is probably not "
						 "what you want to do.");
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
			size_t batch_size = score.get_shape()[0];
			size_t seq_length = score.get_shape()[2];
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
//// Encoder
template <typename F = float>
using EncoderComposer =
	ComposerLayer<MultiheadAttention<F>, PositionWiseFeedForward<F>,
				  LayerNorm<1, F>, LayerNorm<1, F>>;

template <typename F = float> struct Encoder : public EncoderComposer<F> {
		template <Initializer InitWeights, Initializer InitBias>
		Encoder(size_t d_model, size_t num_heads, size_t d_ff, double dropout_p,
				InitWeights init_weights, InitBias init_bias)
			: EncoderComposer<F>(MultiheadAttention<F>(d_model, num_heads,
													   init_weights, init_bias),
								 PositionWiseFeedForward<F>(
									 d_model, d_ff, init_weights, init_bias),
								 LayerNorm<1, F>({d_model}),
								 LayerNorm<1, F>({d_model})),
			  dropout(dropout_p) {}
		Encoder(size_t d_model, size_t num_heads, size_t d_ff, double dropout_p)
			: EncoderComposer<F>(MultiheadAttention<F>(d_model, num_heads),
								 PositionWiseFeedForward<F>(d_model, d_ff),
								 LayerNorm<1, F>({d_model}),
								 LayerNorm<1, F>({d_model})),
			  dropout(dropout_p) {}

		template <typename T, unsigned int n>
		Tensor<LayerHelper::FlintTypeToCpp<transform_type(to_flint_type<T>())>,
			   n>
		forward(Tensor<T, n> &in) {
			auto attn =
				std::get<0>(EncoderComposer<F>::layers).forward(in, in, in);
			auto attn_norm = std::get<2>(EncoderComposer<F>::layers)
								 .forward(in + dropout.forward(attn));
			auto ff =
				std::get<1>(EncoderComposer<F>::layers).forward(attn_norm);
			auto attn_norm2 = std::get<3>(EncoderComposer<F>::layers)
								  .forward(attn_norm + dropout.forward(ff));
			return attn_norm2;
		}

	private:
		Dropout dropout;
};
//// Decoder
template <typename F = float>
using DecoderComposer =
	ComposerLayer<MultiheadAttention<F>, MultiheadAttention<F>,
				  PositionWiseFeedForward<F>, LayerNorm<1, F>, LayerNorm<1, F>,
				  LayerNorm<1, F>>;

template <typename F = float> struct Decoder : public DecoderComposer<F> {

		template <Initializer InitWeights, Initializer InitBias>
		Decoder(size_t d_model, size_t num_heads, size_t d_ff, double dropout_p,
				InitWeights init_weights, InitBias init_bias)
			: EncoderComposer<F>(MultiheadAttention<F>(d_model, num_heads,
													   init_weights, init_bias),
								 MultiheadAttention<F>(d_model, num_heads,
													   init_weights, init_bias),
								 PositionWiseFeedForward<F>(
									 d_model, d_ff, init_weights, init_bias),
								 LayerNorm<1, F>({d_model}),
								 LayerNorm<1, F>({d_model}),
								 LayerNorm<1, F>({d_model})),
			  dropout(dropout_p) {}
		Decoder(size_t d_model, size_t num_heads, size_t d_ff, double dropout_p)
			: EncoderComposer<F>(MultiheadAttention<F>(d_model, num_heads),
								 MultiheadAttention<F>(d_model, num_heads),
								 PositionWiseFeedForward<F>(d_model, d_ff),
								 LayerNorm<1, F>({d_model}),
								 LayerNorm<1, F>({d_model}),
								 LayerNorm<1, F>({d_model})),
			  dropout(dropout_p) {}
		template <typename T, unsigned int n>
		Tensor<LayerHelper::FlintTypeToCpp<transform_type(to_flint_type<T>())>,
			   n>
		forward(Tensor<T, n> &x, Tensor<T, n> &enc_out) {
			auto attn =
				std::get<0>(EncoderComposer<F>::layers).forward(x, x, x);
			auto attn_norm = std::get<3>(EncoderComposer<F>::layers)
								 .forward(x + dropout.forward(attn));
			// cross attention
			auto cross_attn = std::get<1>(EncoderComposer<F>::layers)
								  .forward(attn_norm, enc_out, enc_out);
			auto cross_attn_norm =
				std::get<4>(EncoderComposer<F>::layers)
					.forward(attn_norm + dropout.forward(cross_attn));
			auto ff = std::get<1>(EncoderComposer<F>::layers)
						  .forward(cross_attn_norm);
			auto cross_attn_norm2 =
				std::get<5>(EncoderComposer<F>::layers)
					.forward(cross_attn_norm + dropout.forward(ff));
			return cross_attn_norm2;
		}

	private:
		Dropout dropout;
};
//// Transformer
template <typename F = float> struct Transformer {
		Transformer(size_t src_vocab_size, size_t tgt_vocab_size,
					size_t d_model, size_t num_heads, size_t num_layers,
					size_t d_ff, size_t max_seq_length, double dropout)
			: positional_encoding(d_model, max_seq_length),
			  encoder(num_layers,
					  Encoder<F>(d_model, num_heads, d_ff, dropout)),
			  decoder(num_layers,
					  Decoder<F>(d_model, num_heads, d_ff, dropout)),
			  fc(d_model, tgt_vocab_size), dropout(dropout) {
			// TODO: embeddings
		}
		template <typename T, unsigned int n>
		Tensor<LayerHelper::FlintTypeToCpp<transform_type(to_flint_type<T>())>,
			   n>
		forward(Tensor<T, n> &x, Tensor<T, n> &enc_out) {
			// TODO
		}
		bool training = false;
		// to fulfill generic layer
		template <OptimizerFactory Fac> void generate_optimizer(Fac factory) {
			for (auto &enc : encoder)
				enc.generate_optimizer(factory);
			for (auto &dec : decoder)
				dec.generate_optimizer(factory);
			fc.generate_optimizer(factory);
		}
		// to fulfill generic layer
		template <typename T, unsigned int dim>
		void optimize_weights(const Tensor<T, dim> &error) {
			for (auto &enc : encoder)
				enc.optimize_weights(error);
			for (auto &dec : decoder)
				dec.optimize_weights(error);
			fc.optimize_weights(error);
		}
		void optimize_weights(std::vector<FGraphNode *> grads) {
			for (auto &enc : encoder)
				enc.optimize_weights(grads);
			for (auto &dec : decoder)
				dec.optimize_weights(grads);
			fc.optimize_weights(grads);
		}
		std::vector<FGraphNode *> collect_weights() {
			using namespace std;
			vector<FGraphNode *> res = fc.collect_weights();
			for (auto &enc : encoder) {
				auto weights = enc.collect_weights();
				res.insert(res.end(), weights.begin(), weights.end());
			}
			for (auto &dec : decoder) {
				auto weights = dec.collect_weights();
				res.insert(res.end(), weights);
			}
			return res;
		}
		void set_weights(const std::vector<FGraphNode *> weights) {
			using namespace std;
			const size_t fc_size = fc.collect_weights().size();
			fc.set_weights(weights);
			vector<FGraphNode *> encdecw;
			size_t index = fc_size;
			for (auto &enc : encoder) {
				const size_t enc_size = enc.collect_weights().size();
				for (int i = 0; i < enc_size; i++)
					encdecw.push_back(weights[index++]);
				enc.set_weights(encdecw);
				encdecw.clear();
			}
			for (auto &dec : decoder) {
				const size_t dec_size = dec.collect_weights().size();
				for (int i = 0; i < dec_size; i++)
					encdecw.push_back(weights[index++]);
				dec.set_weights(encdecw);
				encdecw.clear();
			}
		}
		static constexpr FType transform_type(FType t) {
			return higher_type_constexpr(t, to_flint_type<F>());
		}
		static constexpr unsigned int transform_dimensionality(unsigned int n) {
			return n;
		}
		size_t num_parameters() {
			size_t num = fc.num_parameters();
			for (auto &enc : encoder)
				num += enc.num_parameters();
			for (auto &dec : decoder)
				num += dec.num_parameters();
		}
		std::string name() { return "Transformer"; }
		std::string description() {
			return name() + " with " + to_string(encoder.size()) +
				   " encoder/decoder layers";
		}

	private:
		PositionalEncoding positional_encoding;
		std::vector<Encoder<F>> encoder;
		std::vector<Decoder<F>> decoder;
		Connected<F> fc;
		Dropout dropout;
};
#endif
