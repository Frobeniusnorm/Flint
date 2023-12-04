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
#include "index_modification.hpp"
#include "src/operations/implementation.hpp"

template <typename T>
void SliceImpl::unary_expression(T *__restrict__ result,
								 const T *__restrict__ data, size_t from,
								 size_t size, const FGraphNode *curr) {
	FOperation pred = curr->predecessors[0]->operation;
	FSlice *slice = (FSlice *)curr->operation.additional_data;
	std::vector<size_t> acc_sizes = calcAccSizes(curr->operation);
	std::vector<size_t> acc_sizes_pred =
		calcAccSizes(pred.dimensions, pred.shape);
	// calculate start and step size in flattened array
	size_t start = 0;
	for (unsigned int d = 0; d < curr->operation.dimensions; d++) {
		start += slice->start[d] * acc_sizes_pred[d];
	}
	// calculate for each entry corresponding element
	for (size_t i = from; i < from + size; i++) {
		size_t j = start;
		for (unsigned int d = 0; d < curr->operation.dimensions; d++) {
			// get dimension index
			size_t di = (d == 0 ? i : i % acc_sizes[d - 1]) / acc_sizes[d];
			// reproject
			j += di * slice->step[d] * acc_sizes_pred[d];
		}
		result[i] = data[j];
	}
}
void SliceImpl::execute_cpu(const FGraphNode *node,
							std::vector<CPUResultData> predecessor_data,
							void *__restrict__ result, size_t from,
							size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
template <typename T>
void ExtendImpl::unary_expression(T *__restrict__ result,
								  const T *__restrict__ data, size_t from,
								  size_t size, const FGraphNode *curr) {
	FOperation pred = curr->predecessors[0]->operation;
	FExtend *extend = (FExtend *)curr->operation.additional_data;
	std::vector<size_t> acc_sizes = calcAccSizes(curr->operation);
	std::vector<size_t> acc_sizes_pred =
		calcAccSizes(pred.dimensions, pred.shape);
	// calculate for each entry corresponding element
	for (size_t i = from; i < from + size; i++) {
		size_t j = 0;
		bool set_zero = false;
		for (size_t d = 0; d < acc_sizes.size(); d++) {
			long step = extend->step[d];
			bool inv = step < 0;
			if (inv)
				step = -step;
			// get dimension index
			size_t di = (d == 0 ? i : i % acc_sizes[d - 1]) / acc_sizes[d];
			if (di < extend->start[d]) {
				set_zero = true;
				break;
			}
			di -= extend->start[d];
			if (di % step != 0) {
				set_zero = true;
				break;
			}
			di /= step;
			if (di >= pred.shape[d]) {
				set_zero = true;
				break;
			}
			// reverse if negative
			if (inv) {
				di = pred.shape[d] - di - 1;
			}
			// reproject
			j += di * acc_sizes_pred[d];
		}
		result[i] = set_zero ? 0 : data[j];
	}
}
void ExtendImpl::execute_cpu(const FGraphNode *node,
							 std::vector<CPUResultData> predecessor_data,
							 void *__restrict__ result, size_t from,
							 size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
template <typename T, typename A, typename B>
void IndexImpl::binary_expression(T *__restrict__ result,
								  const A *__restrict__ data1,
								  const B *__restrict__ data2, size_t from,
								  size_t size, size_t index_man_1,
								  size_t inv_man_1, size_t index_man_2,
								  size_t inv_man_2, const FGraphNode *curr) {
	const FGraphNode *a = curr->predecessors[0];
	const FGraphNode *b = curr->predecessors[1];
	const FOperation op = curr->operation;
	const unsigned int axis = b->operation.dimensions - 1;
	// get index of result, index tensor, reproject index
	size_t acc_sizes_ax = 1;
	for (int i = axis + 1; i < op.dimensions; i++)
		acc_sizes_ax *= op.shape[i];

	for (size_t i = from; i < from + size; i++) {
		const size_t base = i / (acc_sizes_ax * op.shape[axis]);
		const size_t rest = i % acc_sizes_ax;
		const size_t ind = (size_t)data2[i / acc_sizes_ax];
		result[i] = data1[(base * acc_sizes_ax * a->operation.shape[axis]) +
						  (ind * acc_sizes_ax) + rest];
	}
}
void IndexImpl::execute_cpu(const FGraphNode *node,
							std::vector<CPUResultData> predecessor_data,
							void *__restrict__ result, size_t from,
							size_t size) {
	BINARY_EXECUTE_MONOTON_IMPL
}
template <typename T>
void SetIndexImpl::execute_cpu_typed(
	const FGraphNode *node, std::vector<CPUResultData> predecessor_data,
	T *__restrict__ result, size_t from, size_t size) {
	// TODO Improve (somehow)
	const CPUResultData a = predecessor_data[0];
	const CPUResultData b = predecessor_data[1];
	const CPUResultData c = predecessor_data[2];
	const unsigned int axis = c.shape.size() - 1;
	const FOperation op = node->operation;
	// get index of result, index tensor, reproject index
	size_t acc_sizes_ax = 1;
	for (int i = axis + 1; i < op.dimensions; i++)
		acc_sizes_ax *= op.shape[i];

	for (size_t i = from; i < from + size; i++) {
		const size_t base = i / (acc_sizes_ax * op.shape[axis]);
		const size_t rest = i % acc_sizes_ax;
		const size_t axi = (i / acc_sizes_ax) % op.shape[axis];
		const size_t base_ind = base * c.shape[axis];
		bool found_something = false;
		result[i] = 0;
		// iterate over last dimension and find all correct indices
		for (size_t j = base_ind; j < base_ind + c.shape[axis]; j++) {
			const long ind = (long)(c.type == F_INT32 ? ((int *)c.data)[j]
													  : ((long *)c.data)[j]);
			if (ind == axi) {
				found_something = true;
				result[i] += ((T *)b.data)[j * acc_sizes_ax + rest];
			}
		}
		// if at least one index was found -> only sum of elements of b
		if (!found_something)
			result[i] = ((T *)a.data)[i];
	}
}
void SetIndexImpl::execute_cpu(const FGraphNode *node,
							   std::vector<CPUResultData> predecessor_data,
							   void *__restrict__ result, size_t from,
							   size_t size) {
	EXECUTE_TYPED_IMPL
}
