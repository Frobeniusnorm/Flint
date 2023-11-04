
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
#ifndef FLINT_TRAINER
#define FLINT_TRAINER
#include "losses.hpp"
#include "optimizers.hpp"
#include <flint/flint.h>
#include <flint/flint_helper.hpp>
#include <optional>
// TODO dataloader
template <typename T1, unsigned int n1, typename T2, unsigned int n2>
struct TrainingData {
		Tensor<T1, n1> X;
		Tensor<T2, n2> Y;
		std::optional<Tensor<T1, n1>> vX;
		std::optional<Tensor<T2, n2>> vY;
		TrainingData(Tensor<T1, n1> X, Tensor<T2, n2> Y) : X(X), Y(Y) {}
		TrainingData(Tensor<T1, n1> X, Tensor<T2, n2> Y, Tensor<T1, n1> vX,
					 Tensor<T2, n2> vY)
			: X(X), Y(Y), vX(vX), vY(vY) {}
};

#endif
