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
#include <concepts>
#include "flint.hpp"
#include <limits>
/**
 * Optimizer interface that defines an update method.
 * An optimizer is intended to be instantiated once per weight
 * and optimizes double or flaot weights. 
 * The type-parameter `n` denotes the dimensionality of the
 * weight this optimizer was generated for.*/
template <int n, typename F = float> struct Optimizer {
		virtual ~Optimizer() = default;
		/**
		 * Takes the old weight and its gradient to the error tensor and updates
		 * it, i.e. returns the updated version of the weight.
		 */
		virtual Tensor<F, n> update(Tensor<F, n> &weights,
										 Tensor<F, n> &gradient) = 0;
};
/**
 * An OptimizerFactory is used to generate optimizers on the heap with
 * predefined parameters. Needed so a new optimizer per weight can be generated.
 * For each derivation of `Optimizer` there should be one factory to generate
 * instances of that optimizers for the weights.
 */
template <typename T>
concept OptimizerFactory = requires(T fac) {
	{
		(fac.template generate_optimizer<2>())
	} -> std::convertible_to<Optimizer<2> *>;
	{ fac.name() } -> std::convertible_to<std::string>;
	{ fac.description() } -> std::convertible_to<std::string>;
};
/**
 * Implementation of the Adam algorithm (first-order gradient-based optimizer
 * for stochastic objective functions based on adaptive estimates of lower-order
 * moments).
 */
template <int n, typename F = float> struct Adam : public Optimizer<n, F> {
		const F epsilon = std::numeric_limits<F>::epsilon();
		const F learning_rate, b1, b2;
		/**
		 * Initializes the Adam algorithm with some parameters that influence
		 * the optimization speed and accuracy.
		 *  - `learning_rate`: (sometimes called `alpha`) the step size per
		 *     optimization, i.e. the proportion weights are updated. Higher
		 * values (e.g. 0.2) lead to a faster convergence, while lower values
		 * yield more accurate convergence.
		 *  - `b1`: (sometimes called `beta1`) the exponential decay rate for
		 * the first moment estimates.
		 *  - `b2`: (sometimes called `beta2`) the exponential decay rate for
		 * the second moment estimates.
		 *
		 * You can tune the individual members later on too.
		 */
		Adam(F learning_rate = 0.0015, F b1 = 0.9, F b2 = 0.999)
			: learning_rate(learning_rate), b1(b1), b2(b2) {}
		Tensor<F, n> update(Tensor<F, n> &weight,
								 Tensor<F, n> &grad) {
			if (!init) {
				init = true;
				m = Flint::constant_array((F)0.0, weight.get_shape());
				v = Flint::constant_array((F)0.0, weight.get_shape());
			}
			grad.execute();
			m = m * b1 + grad * (1 - b1);
			v = v * b2 + grad * grad * (1 - b2);
			m.execute();
			v.execute();
			Tensor<F, n> mh = m / (1 - (F)std::pow(b1, t));
			Tensor<F, n> vh = v / (1 - (F)std::pow(b2, t));
			t += 1;
			return weight - (mh * learning_rate) / (vh.sqrt() + epsilon);
		}

	private:
		Tensor<F, n> m;
		Tensor<F, n> v;
		bool init = false;
		unsigned long t = 1;
};
/**
 * Constructs Adam Optimizer with preset parameters.
 */
struct AdamFactory {
		double learning_rate, b1, b2;
		/** Initialisation parameters for the Adam algorithm that influence the
		 * optimization speed and accuracy.
		 *  - `learning_rate`: (sometimes called `alpha`) the step size per
		 *     optimization, i.e. the proportion weights are updated. Higher
		 *     values (e.g. 0.2) lead to a faster convergence, while lower
		 * values yield more accurate convergence.
		 *  - `b1`: (sometimes called `beta1`) the exponential decay rate for
		 *     the first moment estimates.
		 *  - `b2`: (sometimes called `beta2`) the exponential decay rate for
		 *     the second moment estimates. All Adam instances generated by
		 *     `generate_optimizer` are constructed with the given parameters.
		 */
		AdamFactory(double learning_rate = 0.0015, double b1 = 0.9,
					double b2 = 0.999)
			: learning_rate(learning_rate), b1(b1), b2(b2) {}
		/**
		 * Generates an Adam optimizer for a `n`-dimensional weight.
		 */
		template <int n> Optimizer<n> *generate_optimizer() const {
			return new Adam<n>(learning_rate, b1, b2);
		}
		std::string name() const { return "Adam"; }
		std::string description() const {
			return "learning rate: " + std::to_string(learning_rate) +
				   ", b1: " + std::to_string(b1) +
				   ", b2: " + std::to_string(b2);
		}
};

#endif
