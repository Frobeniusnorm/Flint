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
#include "reductions.hpp"
#include <limits>

#define MIN_VAL(x, y) (x < y ? x : y)
#define MAX_VAL(x, y) (x < y ? y : x)

template <typename T>
void ReduceSumImpl::unary_expression(T *__restrict__ result,
									 const T *__restrict__ data, size_t from,
									 size_t size, const FGraphNode *curr) {
	const FOperation pred = curr->predecessors[0]->operation;
	const int dim = ((int *)curr->operation.additional_data)[0];
	size_t it_dim = 1; // iteration size <=> product of all dimensions along dim
	for (size_t d = dim + 1; d < pred.dimensions; d++)
		it_dim *= pred.shape[d];
	for (size_t i = from; i < from + size; i++) {
		result[i] = 0;
		for (size_t j = 0; j < pred.shape[dim]; j++) {
			const T curr = data[(i / it_dim) * it_dim * pred.shape[dim] +
								i % it_dim + j * it_dim];
			result[i] += curr;
		}
	}
}
int ReduceSumImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
									 OCLLazyCodegenState &compiler_state) {}
template <typename T>
void ReduceMulImpl::unary_expression(T *__restrict__ result,
									 const T *__restrict__ data, size_t from,
									 size_t size, const FGraphNode *curr) {
	const FOperation pred = curr->predecessors[0]->operation;
	const int dim = ((int *)curr->operation.additional_data)[0];
	size_t it_dim = 1; // iteration size <=> product of all dimensions along dim
	for (size_t d = dim + 1; d < pred.dimensions; d++)
		it_dim *= pred.shape[d];
	for (size_t i = from; i < from + size; i++) {
		result[i] = 1;
		for (size_t j = 0; j < pred.shape[dim]; j++) {
			const T curr = data[(i / it_dim) * it_dim * pred.shape[dim] +
								i % it_dim + j * it_dim];
			result[i] *= curr;
		}
	}
}
int ReduceMulImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
									 OCLLazyCodegenState &compiler_state) {}
template <typename T>
void ReduceMinImpl::unary_expression(T *__restrict__ result,
									 const T *__restrict__ data, size_t from,
									 size_t size, const FGraphNode *curr) {
	const FOperation pred = curr->predecessors[0]->operation;
	const int dim = ((int *)curr->operation.additional_data)[0];
	size_t it_dim = 1; // iteration size <=> product of all dimensions along dim
	for (size_t d = dim + 1; d < pred.dimensions; d++)
		it_dim *= pred.shape[d];
	for (size_t i = from; i < from + size; i++) {
		result[i] = std::numeric_limits<T>::max();
		for (size_t j = 0; j < pred.shape[dim]; j++) {
			const T curr = data[(i / it_dim) * it_dim * pred.shape[dim] +
								i % it_dim + j * it_dim];
			result[i] = MIN_VAL(result[i], curr);
		}
	}
}
int ReduceMinImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
									 OCLLazyCodegenState &compiler_state) {}
template <typename T>
void ReduceMaxImpl::unary_expression(T *__restrict__ result,
									 const T *__restrict__ data, size_t from,
									 size_t size, const FGraphNode *curr) {
	const FOperation pred = curr->predecessors[0]->operation;
	const int dim = ((int *)curr->operation.additional_data)[0];
	size_t it_dim = 1; // iteration size <=> product of all dimensions along dim
	for (size_t d = dim + 1; d < pred.dimensions; d++)
		it_dim *= pred.shape[d];
	for (size_t i = from; i < from + size; i++) {
		result[i] = std::numeric_limits<T>::lowest();
		for (size_t j = 0; j < pred.shape[dim]; j++) {
			const T curr = data[(i / it_dim) * it_dim * pred.shape[dim] +
								i % it_dim + j * it_dim];
			result[i] = MAX_VAL(result[i], curr);
		}
	}
}
int ReduceMaxImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
									 OCLLazyCodegenState &compiler_state) {}
void ReduceSumImpl::execute_cpu(const FGraphNode *node,
								std::vector<CPUResultData> predecessor_data,
								void *__restrict__ result, size_t from,
								size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
void ReduceMulImpl::execute_cpu(const FGraphNode *node,
								std::vector<CPUResultData> predecessor_data,
								void *__restrict__ result, size_t from,
								size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
void ReduceMinImpl::execute_cpu(const FGraphNode *node,
								std::vector<CPUResultData> predecessor_data,
								void *__restrict__ result, size_t from,
								size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
void ReduceMaxImpl::execute_cpu(const FGraphNode *node,
								std::vector<CPUResultData> predecessor_data,
								void *__restrict__ result, size_t from,
								size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
