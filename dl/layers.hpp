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
#ifndef FLINT_LAYERS
#define FLINT_LAYERS
#include "optimizers.hpp"
#include <bits/utility.h>
#include <concepts>
#include "flint.hpp"
#include "flint_helper.hpp"
#include <memory>
namespace LayerHelper {
/**
 * FOR INTERNAL USE ONLY
 * builds an compile-time linked list of Tensor pointer
 */
template <typename T, unsigned int index, int... w> class WeightRef {};

template <unsigned int index, int n, typename F> struct WeightRef<F, index, n> {
		Tensor<F, n> weight;
		std::unique_ptr<Optimizer<n>> optimizer = nullptr;
		template <unsigned int k, unsigned int f>
		void set_weight(Tensor<F, f> &w) {
			static_assert(k == index, "Invalid weight index!");
			static_assert(f == n,
						  "Could not set weight, wrong number of dimensions! "
						  "Please specify your "
						  "template for the Layer super class correctly!");
			weight = w;
			weight.watch();
		}
		void set_weights(const std::vector<FGraphNode *> nodes) {
			weight = Tensor<F, n>(nodes[index]);
			weight.watch();
		}
		template <OptimizerFactory Fac> void gen_optimizer(const Fac fac) {
			optimizer = std::unique_ptr<Optimizer<n>>(
				fac.template generate_optimizer<n>());
		}
		template <typename T, unsigned int k>
		void optimize(const Tensor<T, k> &error) {
			if (optimizer) {
				Tensor<F, n> gw = error.gradient(weight);
				weight = optimizer->update(weight, gw);
				weight.execute();
				weight.watch();
			} else {
				flogging(F_WARNING, "No Optimizer for weight!");
			}
		}
		template <int i, unsigned int k> Tensor<F, k> &get_weight() {
			static_assert(i == index, "Invalid weight index!");
			return weight;
		}
		void collect_weights(std::vector<FGraphNode *> &nodes) {
			nodes[index] = weight.get_graph_node();
		}
		void update_weights(std::vector<FGraphNode *> &grads) {
			if (optimizer) {
				if (!grads[index])
					return;
				Tensor<F, n> gw(grads[index], weight.get_shape());
				weight = optimizer->update(weight, gw);
				weight.execute();
				weight.watch();
			} else {
				flogging(F_WARNING, "No Optimizer for weight!");
			}
		}
		size_t count_weights() {
			size_t total = 1;
			for (int i = 0; i < n; i++)
				total *= weight.get_shape()[i];
			return total;
		}
};

template <typename F, unsigned int index, int n, int... wn>
struct WeightRef<F, index, n, wn...> {
		Tensor<F, n> weight;
		std::unique_ptr<Optimizer<n>> optimizer = nullptr;
		WeightRef<F, index + 1, wn...> others;
		template <unsigned int k, unsigned int f>
		void set_weight(Tensor<F, f> &w) {
			if constexpr (k == index) {
				static_assert(
					f == n, "Could not set weight, wrong number of dimensions! "
							"Please specify your "
							"template for the Layer super class correctly!");
				weight = w;
				weight.watch();
			} else
				others.template set_weight<k>(w);
		}
		void set_weights(const std::vector<FGraphNode *> nodes) {
			weight = Tensor<F, n>(nodes[index]);
			weight.watch();
			others.template set_weights(nodes);
		}
		template <OptimizerFactory Fac> void gen_optimizer(const Fac fac) {
			optimizer = std::unique_ptr<Optimizer<n>>(
				fac.template generate_optimizer<n>());
			others.gen_optimizer(fac);
		}
		template <typename T, unsigned int k>
		void optimize(const Tensor<T, k> &error) {
			if (optimizer) {
				Tensor<F, n> gw = error.gradient(weight);
				weight = optimizer->update(weight, gw);
				weight.execute();
				weight.watch();
			} else {
				flogging(F_WARNING, "No Optimizer for weight!");
			}
			others.optimize(error);
		}
		template <int i, unsigned int k> Tensor<F, k> &get_weight() {
			if constexpr (i == index)
				return weight;
			else
				return others.template get_weight<i, k>();
		}
		void collect_weights(std::vector<FGraphNode *> &nodes) {
			nodes[index] = weight.get_graph_node();
			others.collect_weights(nodes);
		}
		void update_weights(std::vector<FGraphNode *> &grads) {
			if (optimizer) {
				if (grads[index]) {
					Tensor<F, n> gw(grads[index], weight.get_shape());
					weight = optimizer->update(weight, gw);
					weight.execute();
					weight.watch();
				}
			} else {
				flogging(F_WARNING, "No Optimizer for weight!");
			}
			others.update_weights(grads);
		}
		size_t count_weights() {
			size_t total = 1;
			for (int i = 0; i < n; i++)
				total *= weight.get_shape()[i];
			return total + others.count_weights();
		}
};
template <FType t>
using FlintTypeToCpp = typename std::conditional<
	t == F_INT32, int,
	typename std::conditional<
		t == F_INT64, long,
		typename std::conditional<t == F_FLOAT32, float, double>::type>::type>::
	type;
} // namespace LayerHelper
/**
 * Concept of methods a Layer for neural networks has to implement.
 * Allows a modular forward method and therefore no type safety for it.
 * Used internally in other layers to allow higher flexibility than the
 * `GenericLayer` concept.
 */
template <typename T>
concept GenericModule =
	requires(T a, Tensor<float, 2> &t1, Tensor<int, 2> &t2,
			 Tensor<double, 2> &t3, Tensor<long, 2> &t4, AdamFactory fac,
			 std::vector<FGraphNode *> grads) {
		a.optimize_weights(t1);
		a.optimize_weights(t2);
		a.optimize_weights(t3);
		a.optimize_weights(t4);
		{
			a.collect_weights()
		} -> std::convertible_to<std::vector<FGraphNode *>>;
		a.optimize_weights(grads);
		a.set_weights(grads);
		a.generate_optimizer(fac);
		a.training = true;
		{ T::transform_dimensionality(5) } -> std::convertible_to<unsigned int>;
		{ T::transform_type(F_INT32) } -> std::convertible_to<FType>;
		{ a.name() } -> std::convertible_to<std::string>;
		{ a.description() } -> std::convertible_to<std::string>;
		{ a.num_parameters() } -> std::convertible_to<size_t>;
	};
/**
 * Concept of methods a Layer for neural networks has to implement.
 * Mind the static constexpr methods that determine the modifications of
 * dimensionality and types of the input tensors `int
 * transform_dimensionality(int)` and `FType transform_type(FType)`, they
 * describe the type of your forward (i.e. if a tensor of dimensionality `n` and
 * type `T` is inserted into your forward, a tensor of dimensionality
 * `transform_dimensionality(n)` and type `transform_type(T)` should be
 * returned). It is highly recommended to derive your Layer from
 * `UntrainableLayer` or `Layer`, since they provide already implementations for
 * some methods. `forward` may consume its input tensor since it isn't needed
 * afterwards.
 */
template <typename T>
concept GenericLayer =
	requires(T a, Tensor<float, 2> &t1, Tensor<int, 2> &t2,
			 Tensor<double, 2> &t3, Tensor<long, 2> &t4, AdamFactory fac,
			 std::vector<FGraphNode *> grads) {
		{
			a.forward(t1)
		} -> std::convertible_to<
			Tensor<LayerHelper::FlintTypeToCpp<T::transform_type(F_FLOAT32)>,
				   T::transform_dimensionality(2)>>;
		{
			a.forward(t2)
		} -> std::convertible_to<
			Tensor<LayerHelper::FlintTypeToCpp<T::transform_type(F_INT32)>,
				   T::transform_dimensionality(2)>>;
		{
			a.forward(t3)
		} -> std::convertible_to<
			Tensor<LayerHelper::FlintTypeToCpp<T::transform_type(F_FLOAT64)>,
				   T::transform_dimensionality(2)>>;
		{
			a.forward(t4)
		} -> std::convertible_to<
			Tensor<LayerHelper::FlintTypeToCpp<T::transform_type(F_INT64)>,
				   T::transform_dimensionality(2)>>;
		GenericModule<T>;
	};
/**
 * Implements blank methods for every method of GenericLayer that is not needed
 * for a Layer that is not trainable.
 * If you derive from this class you have to implement the `forward` method from
 * the `GenericLayer` concept and - if the forward outputs another type or
 * dimensionality then its parameter has - overload `transform_type` and
 * `transform_dimensionality`.
 */
struct UntrainableLayer {
		bool training = false;
		// to fulfill generic layer
		template <OptimizerFactory Fac> void generate_optimizer(Fac factory) {}
		// to fulfill generic layer
		template <typename T, unsigned int dim>
		void optimize_weights(const Tensor<T, dim> &error) {}
		void optimize_weights(std::vector<FGraphNode *> grads) {}
		std::vector<FGraphNode *> collect_weights() { return {}; }
		void set_weights(const std::vector<FGraphNode *> weights) {}
		static constexpr FType transform_type(FType t) { return t; }
		static constexpr unsigned int transform_dimensionality(unsigned int n) {
			return n;
		}
		size_t num_parameters() { return 0; }
		virtual std::string name() { return "unnamed"; }
		virtual std::string description() { return name() + " layer"; }
};
/**
 * Virtual super class for Layers that are composed of other Layers.
 * The managed Layers are given in the constructor and their types in
 * the variadic template. Most methods of the GenericLayer are already
 * implemented, the forward method still has to be defined.
 */
template <GenericModule... LayerTypes> struct ComposerLayer {
		std::tuple<LayerTypes...> layers;
		ComposerLayer(LayerTypes... layers) : layers(layers...) {}
		bool training = false;
		// to fulfill generic layer
		template <OptimizerFactory Fac, int l = 0>
		void generate_optimizer(Fac factory) {
			std::get<l>(layers).generate_optimizer(factory);
			if constexpr (l < std::tuple_size<decltype(layers)>::value - 1)
				generate_optimizer<l + 1>(factory);
		}
		// to fulfill generic layer
		template <typename T, unsigned int dim, int l = 0>
		void optimize_weights(const Tensor<T, dim> &error) {
			std::get<l>(layers).optimize_weights(error);
			if constexpr (l < std::tuple_size<decltype(layers)>::value - 1)
				optimize_weights<T, dim, l + 1>(error);
		}
		template <int l = 0>
		void optimize_weights(std::vector<FGraphNode *> grads) {
			std::get<l>(layers).optimize_weights(grads);
			if constexpr (l < std::tuple_size<decltype(layers)>::value - 1)
				optimize_weights<l + 1>(grads);
		}
		template <int l = 0> std::vector<FGraphNode *> collect_weights() {
			auto weights = std::get<l>(layers).collect_weights();
			if constexpr (l < std::tuple_size<decltype(layers)>::value - 1) {
				auto other_weights = collect_weights<l + 1>();
				weights.insert(weights.end(), other_weights.begin(),
							   other_weights.end());
			}
			return weights;
		}
		template <int l = 0>
		void set_weights(const std::vector<FGraphNode *> weights) {
			size_t num_weights = std::get<l>(layers).collect_weights().size();
			std::vector<FGraphNode *> curr(num_weights);
			std::vector<FGraphNode *> next(weights.size() - num_weights);
			for (int i = 0; i < num_weights; i++)
				curr[i] = weights[i];
			for (int i = 0; i < next.size(); i++)
				next[i] = weights[i + num_weights];
			std::get<l>(layers).set_weights(curr);
			set_weights<l + 1>(next);
		}
		static constexpr FType transform_type(FType t) { return t; }
		static constexpr unsigned int transform_dimensionality(unsigned int n) {
			return n;
		}
		template <int l = 0> size_t num_parameters() {
			if constexpr (l == std::tuple_size<decltype(layers)>::value - 1)
				return std::get<l>(layers).num_parameters();
			else
				return std::get<l>(layers).num_parameters() +
					   num_parameters<l + 1>();
		}
		virtual std::string name() { return "unnamed"; }
		virtual std::string description() { return name() + " layer"; }
};
/**
 * Virtual super class of all Layer implementations with type safe weight
 * management capabilities. The variadic template describes the dimensionality
 * of the individual weights i.e. a `Layer<double, 3,4,5>` has three weights:
 * `Tensor<double, 3>`, `Tensor<double, 4>`, `Tensor<double, 5>`.
 * You have to initialize them by providing their initial state in the
 * constructor, after that you may access references to them with the function
 * `get_weight<int index>()`.
 *
 * If you derive from this class you have to implement the `forward` method from
 * the `GenericLayer` concept and - if the `forward` outputs another type or
 * dimensionality then its parameter has - overload `transform_type` and
 * `transform_dimensionality`.
 */
template <typename F, int... wn> class Layer {
	protected:
		LayerHelper::WeightRef<F, 0, wn...> weight_refs;
		template <unsigned int index, unsigned int n>
		void init_weights(Tensor<F, n> &t) {
			weight_refs.template set_weight<index, n>(t);
		}
		template <unsigned int index, unsigned int n, typename... args>
		void init_weights(Tensor<F, n> &t, args &...weights) {
			weight_refs.template set_weight<index, n>(t);
			init_weights<index + 1>(weights...);
		}
		template <int index, int w, int... wo> static constexpr int get_dim() {
			if constexpr (index == 0)
				return w;
			else
				return get_dim<index - 1, wo...>();
		}

	public:
		bool training = false;
		static constexpr FType transform_type(FType t) {
			return higher_type_constexpr(t, to_flint_type<F>());
		}
		static constexpr unsigned int transform_dimensionality(unsigned int n) {
			return n;
		}
		Layer() = default;
		/** Initializes the weights by copying the provided ones.
		 * After that you may access them with `get_weight<int index>()`. */
		template <typename... args> Layer(args... weights) {
			init_weights<0>(weights...);
		}
		/** Sets a specific weight described by its index */
		template <int index, int dim> void set_weight(Tensor<F, dim> t) {
			weight_refs.template set_weight<index>(std::move(t));
		}
		/** Sets all weights from an array */
		void set_weights(const std::vector<FGraphNode *> weights) {
			weight_refs.set_weights(weights);
		}
		/** Returns a reference to a specific weight described by its index */
		template <int index> Tensor<F, get_dim<index, wn...>()> &get_weight() {
			return weight_refs
				.template get_weight<index, get_dim<index, wn...>()>();
		}
		/** Creates an optimizer for each weight with the methods of the
		 * provided `OptimizerFactory` */
		template <OptimizerFactory Fac> void generate_optimizer(Fac factory) {
			weight_refs.gen_optimizer(factory);
		}
		/** Calculates the gradients of each weight to the `error` tensor and
		 * optimizes them by their gradient with their optimizer (if one has
		 * been generated, see `generate_optimizer()`) */
		template <typename T, unsigned int dim>
		void optimize_weights(const Tensor<T, dim> &error) {
			weight_refs.optimize(error);
		}
		/** Collects pointer to the underlying `FGraphNode` references of the
		 * weights. Usefull for gradient calculation. */
		std::vector<FGraphNode *> collect_weights() {
			std::vector<FGraphNode *> nodes(sizeof...(wn));
			weight_refs.collect_weights(nodes);
			return nodes;
		}
		/** Takes already calculated Gradients of the weights (`ǹ`th entry in
		 * `grads` correspons to the `n`th weight) and optimizes them by their
		 * gradient with their optimizer (if one has been generated, see
		 * `generate_optimizer()`) */
		void optimize_weights(std::vector<FGraphNode *> grads) {
			weight_refs.update_weights(grads);
		}
		/** Returns the name of this Layer for overviews and debugging. */
		virtual std::string name() { return "unnamed"; }
		/** Returns a summary of this Layer for overviews and debugging. */
		virtual std::string description() { return name() + " layer"; }
		size_t num_parameters() { return weight_refs.count_weights(); }
};
#endif
