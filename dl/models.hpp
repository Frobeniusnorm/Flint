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
#include "layers.hpp"
#include "losses.hpp"
#include "optimizers.hpp"
#include <chrono>
#include <flint/flint.h>
#include <flint/flint_helper.hpp>
#include <iomanip>
#include <list>
#include <math.h>
#include <memory>
#include <tuple>
#include <vector>

template <FType in> constexpr FType get_output_type() { return in; }
template <FType in, GenericLayer K> constexpr FType get_output_type() {
	return K::transform_type(in);
}
template <FType in, GenericLayer K1, GenericLayer K2, GenericLayer... F>
constexpr FType get_output_type() {
	constexpr FType out = K2::transform_type(K1::transform_type(in));
	return get_output_type<out, F...>();
}
template <unsigned int in> constexpr unsigned int get_output_dim() {
	return in;
}
template <unsigned int in, GenericLayer K>
constexpr unsigned int get_output_dim() {
	return K::transform_dimensionality(in);
}
template <unsigned int in, GenericLayer K1, GenericLayer K2, GenericLayer... F>
constexpr unsigned int get_output_dim() {
	constexpr unsigned int out =
		K2::transform_dimensionality(K1::transform_dimensionality(in));
	return get_output_dim<out, F...>();
}
template <unsigned int layer, unsigned int in, unsigned int curr,
		  GenericLayer K>
constexpr unsigned int get_layer_dim() {
	static_assert(layer == curr, "Could not deduce Layer dimensionality!");
	return K::transform_dimensionality(in);
}
template <unsigned int layer, unsigned int in, unsigned int curr,
		  GenericLayer K, GenericLayer... F>
constexpr unsigned int get_layer_dim() {
	constexpr unsigned int out = K::transform_dimensionality(in);
	if constexpr (layer == curr)
		return out;
	else
		return get_layer_dim<layer, out, curr + 1, F...>();
}
template <unsigned int layer, FType in, unsigned int curr, GenericLayer K>
constexpr FType get_layer_type() {
	static_assert(layer == curr, "Could not deduce Layer dimensionality!");
	return K::transform_type(in);
}
template <unsigned int layer, FType in, unsigned int curr, GenericLayer K,
		  GenericLayer... F>
constexpr FType get_layer_type() {
	constexpr FType out = K::transform_type(in);
	if constexpr (layer == curr)
		return out;
	else
		return get_layer_type<layer, out, curr + 1, F...>();
}
/**
 * Model where each layer outputs the input of the next layer.
 * Best used with C++ auto typing:
 *
 * @code{
 * auto model = SequentialModel(
 *  Connected(10, 20),
 *  Relu(),
 *  Dropout(0.1),
 *  Connected(20, 10),
 *  SoftMax()
 * ); // has type SequentialModel<Connected, Relu, Dropout, Connected, SoftMax>
 * }
 */
template <GenericLayer... T> struct SequentialModel {
		std::tuple<T...> layers;
		SequentialModel(T... layers) : layers(std::move(layers)...) {}

		template <OptimizerFactory Fac> void generate_optimizer(Fac fac) {
			gen_opt<0>(fac);
		}
		/**
		 * Passes a batch of input tensors through all layers and returns the
		 * output of the last layer.
		 */
		template <typename K, unsigned int n>
		Tensor<LayerHelper::FlintTypeToCpp<
				   get_output_type<toFlintType<K>(), T...>()>,
			   get_output_dim<n, T...>()>
		forward_batch(Tensor<K, n> &in) {
			in.get_graph_node()->reference_counter++;
			auto out =
				forward_helper<0,
							   LayerHelper::FlintTypeToCpp<
								   get_output_type<toFlintType<K>(), T...>()>,
							   get_output_dim<n, T...>(), K, n>(
					in.get_graph_node());
			return out;
		}
		/**
		 * Passes an input tensor through all layers and returns the output of
		 * the last layer.
		 */
		template <typename K, unsigned int n>
		Tensor<LayerHelper::FlintTypeToCpp<
				   get_output_type<toFlintType<K>(), T...>()>,
			   get_output_dim<n, T...>()>
		forward(Tensor<K, n> &in) {
			// because layers expect batches
			Tensor<K, n + 1> expanded = in.expand(0, 1);
			std::cout << expanded << std::endl;
			expanded.get_graph_node()->reference_counter++;
			auto out =
				forward_helper<0,
							   LayerHelper::FlintTypeToCpp<
								   get_output_type<toFlintType<K>(), T...>()>,
							   get_output_dim<n + 1, T...>(), K, n + 1>(
					expanded.get_graph_node());
			return out;
		}
		/**
		 * Optimizes the weights (calculates the gradients + calls the
		 * optimizers) of all layer to an error.
		 */
		template <typename K, unsigned int n>
		void optimize(const Tensor<K, n> &error) {
			backward<0>(error);
		}
		void load(const std::string path) {
			using namespace std;
			ifstream file(path, ios::binary);
			vector<char> data;
			{
				list<char> file_cont;
				while (!file.eof())
					file_cont.push_back(file.get());
				file.close();
				data = vector<char>(file_cont.begin(), file_cont.end());
			}
			// collect weights to know how many weights each layer has
			std::vector<std::vector<FGraphNode *>> vars;
			collect_weights<0>(vars);
			size_t index = 0;
			for (int i = 0; i < vars.size(); i++)
				for (int j = 0; j < vars[i].size(); j++) {
					size_t read;
					vars[i][j] = fdeserialize(data.data() + index, &read);
					index += read;
				}
			set_weights<0>(vars);
			flogging(F_VERBOSE,
					 "loaded weights, " + to_string(index) + " bytes");
		}
		void save(const std::string path) {
			using namespace std;
			ofstream file(path, ios::binary);
			std::vector<std::vector<FGraphNode *>> vars;
			collect_weights<0>(vars);
			for (const vector<FGraphNode *> layer : vars) {
				for (FGraphNode *weight : layer) {
					size_t length;
					const char *data = fserialize(weight, &length);
					file.write(data, length);
				}
			}
			file.close();
			flogging(F_VERBOSE, "stored weights");
		}
		template <typename T1, unsigned int n1>
		void backward(Tensor<T1, n1> &error) {
			std::vector<std::vector<FGraphNode *>> vars;
			collect_weights<0>(vars);
			std::vector<FGraphNode *> flat_vars;
			for (unsigned int i = 0; i < vars.size(); i++)
				flat_vars.insert(flat_vars.end(), vars[i].begin(),
								 vars[i].end());
			std::vector<FGraphNode *> grads(flat_vars.size());
			// calculate gradients
			fCalculateGradients(error.get_graph_node(), flat_vars.data(),
								flat_vars.size(), grads.data());
			// reconstruct for layers
			std::vector<std::vector<FGraphNode *>> plgrads(vars.size());
			int index = 0;
			for (unsigned int i = 0; i < vars.size(); i++) {
				plgrads[i] = std::vector<FGraphNode *>(vars[i].size());
				for (unsigned int j = 0; j < vars[i].size(); j++) {
					FGraphNode *curr_grad = grads[index++];
					plgrads[i][j] =
						curr_grad ? fOptimizeMemory(fExecuteGraph(curr_grad))
								  : nullptr;
				}
			}
			backward<0>(plgrads);
		}
		void enable_training() { set_training<0>(true); }
		void disable_training() { set_training<0>(false); }
		/** Returns a small summary of the model. */
		std::string summary() { return summary_helper<0>(); }
		/**
		 * Returns a per-layer vector of all weight-tensors of that layer
		 */
		std::vector<std::vector<FGraphNode *>> collect_weights() {
			std::vector<std::vector<FGraphNode *>> vars;
			collect_weights<0>(vars);
			return vars;
		}

	private:
		template <int n, typename K, unsigned int k>
		inline void backward(const Tensor<K, k> &error) {
			if constexpr (n < sizeof...(T)) {
				std::get<n>(layers).optimize_weights(error);
				backward<n + 1>(error);
			}
		}
		template <int n>
		inline void
		backward(const std::vector<std::vector<FGraphNode *>> grads) {
			if constexpr (n < sizeof...(T)) {
				std::get<n>(layers).optimize_weights(grads[n]);
				backward<n + 1>(grads);
			}
		}
		template <int n>
		inline void
		set_weights(const std::vector<std::vector<FGraphNode *>> weights) {
			if constexpr (n < sizeof...(T)) {
				std::get<n>(layers).set_weights(weights[n]);
				set_weights<n + 1>(weights);
			}
		}
		template <int n, OptimizerFactory Fac> inline void gen_opt(Fac fac) {
			if constexpr (n < sizeof...(T)) {
				std::get<n>(layers).generate_optimizer(fac);
				gen_opt<n + 1>(fac);
			}
		}
		template <int n> inline void set_training(bool b) {
			if constexpr (n < sizeof...(T)) {
				std::get<n>(layers).training = b;
				set_training<n + 1>(b);
			}
		}
		template <int n> std::string summary_helper() {
			if constexpr (n < sizeof...(T))
				return std::to_string(n + 1) + ". " +
					   std::get<n>(layers).summary() + "\n" +
					   summary_helper<n + 1>();
			return "";
		}
		template <int n>
		inline void
		collect_weights(std::vector<std::vector<FGraphNode *>> &vars) {
			if constexpr (n < sizeof...(T)) {
				vars.push_back(std::get<n>(layers).collect_weights());
				collect_weights<n + 1>(vars);
			}
		}
		template <int layer, typename T2, unsigned int n2, typename T1,
				  unsigned int n1>
		inline Tensor<T2, n2> forward_helper(FGraphNode *in) {
			FGraphNode *out;
			{
				Tensor<T1, n1> it(in);
				// now in is no longer needed (reference counter has been
				// artifically incremented) will be freed with it at the end of
				// the block
				in->reference_counter--;
				auto ot = std::get<layer>(layers).forward(it);
				// out is still needed -> save the GraphNode handle from
				// destruction with
				out = ot.get_graph_node();
				out->reference_counter++;
			}
			if constexpr (layer == sizeof...(T) - 1) {
				// will be hold by the Tensor
				out->reference_counter--;
				return Tensor<T2, n2>(out);
			} else {
				return forward_helper<
					layer + 1, T2, n2,
					LayerHelper::FlintTypeToCpp<
						get_layer_type<layer, toFlintType<T1>(), 0, T...>()>,
					get_layer_dim<layer, n1, 0, T...>()>(out);
			}
		}
};
#endif
