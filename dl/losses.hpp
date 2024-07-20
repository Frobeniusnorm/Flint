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
#ifndef FLINT_LOSSES
#define FLINT_LOSSES
#include <concepts>
#include "flint.hpp"
/**
 * Defines the general concept of a Loss function.
 * It receives two tensors: the actual output and the expected one.
 * It then calculates the loss as a double Tensor (since the weights are always
 * double Tensors as well).
 */
template <typename T=float>
concept GenericLoss = requires(T a, Tensor<float, 2> &t1, Tensor<int, 2> &t2,
							   Tensor<double, 2> &t3, Tensor<long, 2> &t4) {
	{
		a.calculate_error(t1, t1)
	} -> std::convertible_to<
		Tensor<float,
			   T::transform_dimensionality(2)>>;
	{
		a.calculate_error(t2, t2)
	} -> std::convertible_to<
		Tensor<double,
			   T::transform_dimensionality(2)>>;
	{
		a.calculate_error(t3, t3)
	} -> std::convertible_to<
		Tensor<double,
			   T::transform_dimensionality(2)>>;
	{
		a.calculate_error(t4, t4)
	} -> std::convertible_to<
		Tensor<double,
			   T::transform_dimensionality(2)>>;
	{ a.name() } -> std::convertible_to<std::string>;
};

/** Calculates the Categorical Cross Entropy Loss with full summation. It is
 * advised to apply a softmax as the last activation layer in the calculation of
 * `in`.
 *
 * Calculates: `sum(-expected * log(in))`
 * */
 struct CrossEntropyLoss {
		CrossEntropyLoss() {
		}
		static constexpr unsigned int transform_dimensionality(int n) {
			return 1;
		}
		template <typename T, unsigned int n>
		Tensor<
			to_float<T>,
			transform_dimensionality(n)>
		calculate_error(Tensor<T, n> &in, Tensor<T, n> &expected) {
			auto pred =
				(in / in.reduce_sum(n - 1).expand(n - 1, in.get_shape()[n - 1]))
					.max((to_float<T>)1e-7)
					.min((to_float<T>)(1 - 1e-7));
			auto t1 = (expected * -pred.log()).reduce_sum();
			size_t total_size = 1;
			for (unsigned int i = 0; i < n - 1; i++)
				total_size *= in.get_shape()[i];
			return (t1 / (to_float<T>)total_size);
		}
		std::string name() { return "Cross Entropy Loss"; }
};
#endif
