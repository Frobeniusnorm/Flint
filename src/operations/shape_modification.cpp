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
#include "shape_modification.hpp"
#include "src/backend_cpu/cpu_common.hpp"
#include "src/operations/implementation.hpp"
#include <cstring>

void FlattenImpl::execute_cpu(const FGraphNode *node,
							  std::vector<CPUResultData> predecessor_data,
							  void *__restrict__ result, size_t from,
							  size_t size) {
	const CPUResultData pred = predecessor_data[0];
	switch (predecessor_data[0].type) {
	case F_INT32:
	case F_FLOAT32:
		std::memcpy((int *)result + from, (int *)pred.data + from, 4 * size);
		break;
	case F_INT64:
	case F_FLOAT64:
		std::memcpy((long *)result + from, (long *)pred.data + from, 8 * size);
		break;
	}
}
template <typename T, typename A>
void ConversionImpl::unary_expression(T *__restrict__ result,
									  const A *__restrict__ data1, size_t from,
									  size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = (T)data1[i];
}
void ConversionImpl::execute_cpu(const FGraphNode *node,
								 std::vector<CPUResultData> predecessor_data,
								 void *__restrict__ result, size_t from,
								 size_t size) {
	UNARY_EXECUTE_IMPL
}
template <typename T>
void RepeatImpl::unary_expression(T *__restrict__ result,
								  const T *__restrict__ data, size_t from,
								  size_t size, const FGraphNode *curr) {
	const FOperation op = curr->operation;
	const FOperation pred = curr->predecessors[0]->operation;
	// calculate number of elements per dimension entry for destination and
	// source
	std::vector<size_t> acc_sizes_d = calcAccSizes(op);
	std::vector<size_t> acc_sizes_s = calcAccSizes(pred.dimensions, pred.shape);
	for (int i = from; i < from + size; i++) {
		// to get the index in the source array we first calculate the
		// indices and reproject
		size_t index = i;
		size_t src_index = 0;
		for (int dim = 0; dim < op.dimensions; dim++) {
			int curr_idx = index / acc_sizes_d[dim];
			index %= acc_sizes_d[dim];
			src_index += (curr_idx % pred.shape[dim]) * acc_sizes_s[dim];
		}
		result[i] = data[src_index];
	}
}
void RepeatImpl::execute_cpu(const FGraphNode *node,
							 std::vector<CPUResultData> predecessor_data,
							 void *__restrict__ result, size_t from,
							 size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
template <typename T>
void TransposeImpl::unary_expression(T *__restrict__ result,
									 const T *__restrict__ data, size_t from,
									 size_t size, const FGraphNode *curr) {
	const FOperation op = curr->operation;
	const int *transposition = (int *)op.additional_data;
	const FOperation pred = curr->predecessors[0]->operation;
	// calculate number of elements per dimension entry for destination and
	// source
	std::vector<size_t> acc_sizes_d = calcAccSizes(op);
	std::vector<size_t> acc_sizes_s = calcAccSizes(pred.dimensions, pred.shape);
	for (int i = from; i < from + size; i++) {
		// to get the index in the source array we first calculate the
		// indices and reproject
		int index = i;
		int src_index = 0;
		for (int dim = 0; dim < op.dimensions; dim++) {
			int curr_idx = index / acc_sizes_d[dim];
			index %= acc_sizes_d[dim];
			src_index += curr_idx * acc_sizes_s[transposition[dim]];
		}
		result[i] = data[src_index];
	}
}
void TransposeImpl::execute_cpu(const FGraphNode *node,
								std::vector<CPUResultData> predecessor_data,
								void *__restrict__ result, size_t from,
								size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
template <typename T>
void ConcatImpl::binary_expression(T *__restrict__ result,
								   const T *__restrict__ data1,
								   const T *__restrict__ data2, size_t from,
								   size_t size, size_t index_man_1,
								   size_t inv_man_1, size_t index_man_2,
								   size_t inv_man_2, const FGraphNode *curr) {
	FGraphNode *a = curr->predecessors[0];
	FGraphNode *b = curr->predecessors[1];
	unsigned int ax = ((unsigned int *)curr->operation.additional_data)[0];
	size_t acc_size_last = 1;
	for (int i = curr->operation.dimensions - 2; i >= (int)ax; i--) {
		acc_size_last *= curr->operation.shape[i + 1];
	}
	for (size_t index = from; index < from + size; index++) {
		size_t sx = index / acc_size_last;
		size_t sc = ax > 0 ? sx % curr->operation.shape[ax] : sx;
		// reproject to one of the tensors
		if (sc < a->operation.shape[ax]) {
			size_t ai = (sx / curr->operation.shape[ax]) * acc_size_last *
							a->operation.shape[ax] +
						sc * acc_size_last + index % acc_size_last;
			result[index] = data1[ai];
		} else {
			size_t bi = (sx / curr->operation.shape[ax]) * acc_size_last *
							b->operation.shape[ax] +
						(sc - a->operation.shape[ax]) * acc_size_last +
						index % acc_size_last;
			result[index] = data2[bi];
		}
	}
}
void ConcatImpl::execute_cpu(const FGraphNode *node,
							 std::vector<CPUResultData> predecessor_data,
							 void *__restrict__ result, size_t from,
							 size_t size) {
	BINARY_EXECUTE_MONOTON_IMPL
}
