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
#include <cstring>

using namespace std;

void FlattenImpl::execute_cpu(const FGraphNode *node,
							  vector<CPUResultData> predecessor_data,
							  void *__restrict__ result, size_t from,
							  size_t size) {
	const CPUResultData pred = predecessor_data[0];
	switch (predecessor_data[0].type) {
	case F_INT32:
	case F_FLOAT32:
		memcpy((int *)result + from, (int *)pred.data + from, 4 * size);
		break;
	case F_INT64:
	case F_FLOAT64:
		memcpy((long *)result + from, (long *)pred.data + from, 8 * size);
		break;
	}
}
int FlattenImpl::generate_ocl_lazy(const FGraphNode *node, string name,
								   OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + typeString(node->operation.data_type) + " " + name + " = v" +
		to_string(compiler_state.variable_index + 1) + ";\n");
	return 0;
}
template <typename T, typename A>
void ConversionImpl::unary_expression(T *__restrict__ result,
									  const A *__restrict__ data1, size_t from,
									  size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = (T)data1[i];
}
int ConversionImpl::generate_ocl_lazy(const FGraphNode *node, string name,
									  OCLLazyCodegenState &compiler_state) {
	const string type = typeString(node->operation.data_type);
	compiler_state.code.prepend(
		"const " + type + " " + name + " = (" + type + ")v" +
		to_string(compiler_state.variable_index + 1) + ";\n");
	return 0;
}
void ConversionImpl::execute_cpu(const FGraphNode *node,
								 vector<CPUResultData> predecessor_data,
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
	vector<size_t> acc_sizes_d = calcAccSizes(op);
	vector<size_t> acc_sizes_s = calcAccSizes(pred.dimensions, pred.shape);
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
int RepeatImpl::generate_ocl_lazy(const FGraphNode *node, string name,
								  OCLLazyCodegenState &compiler_state) {
	const FOperation op = node->operation;
	const FOperation pred = node->predecessors[0]->operation;
	const unsigned int old_idx = compiler_state.num_indices++;
	Twine index_defs;
	index_defs += "int old_index" + to_string(old_idx) + " = index;\n";
	// add to index_defs a redefinition of index, so that we remap
	// to src data calculate number of elements per dimension entry
	// for destination and source
	std::vector<size_t> acc_sizes_d(op.dimensions);
	std::vector<size_t> acc_sizes_s(op.dimensions);
	acc_sizes_d[op.dimensions - 1] = 1;
	acc_sizes_s[op.dimensions - 1] = 1;
	for (int dim = op.dimensions - 2; dim >= 0; dim--) {
		acc_sizes_d[dim] = acc_sizes_d[dim + 1] * op.shape[dim + 1];
		acc_sizes_s[dim] = acc_sizes_s[dim + 1] * pred.shape[dim + 1];
	}
	// to get the index in the source array we first calculate the
	// indices and reproject
	index_defs += "{\nint working_index = index;\nindex = 0;\n";
	for (int dim = 0; dim < op.dimensions; dim++) {
		index_defs += "index += ((working_index /" +
					  to_string(acc_sizes_d[dim]) + ") % " +
					  to_string(pred.shape[dim]) + ") * " +
					  to_string(acc_sizes_s[dim]) + ";\n";
		index_defs += "working_index %= " + to_string(acc_sizes_d[dim]) + ";\n";
	}
	index_defs += "}\n";
	compiler_state.index_defs = index_defs;
	compiler_state.code.prepend("index = old_index" + to_string(old_idx) +
								";\n");
	compiler_state.code.prepend(
		"const " + typeString(node->operation.data_type) + " " + name + " = v" +
		to_string(compiler_state.variable_index + 1) + ";\n");
	return 0;
}
void RepeatImpl::execute_cpu(const FGraphNode *node,
							 vector<CPUResultData> predecessor_data,
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
	vector<size_t> acc_sizes_d = calcAccSizes(op);
	vector<size_t> acc_sizes_s = calcAccSizes(pred.dimensions, pred.shape);
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
int TransposeImpl::generate_ocl_lazy(const FGraphNode *node, string name,
									 OCLLazyCodegenState &compiler_state) {
	const FOperation op = node->operation;
	const int *transposition = (int *)op.additional_data;
	const FOperation pred = node->predecessors[0]->operation;
	unsigned int old_idx = compiler_state.num_indices++;
	Twine index_defs;
	index_defs += "long old_index" + to_string(old_idx) + " = index;\n";
	// add to index_defs a redefinition of index, so that we remap
	// to src data calculate number of elements per dimension entry
	// for destination and source
	std::vector<size_t> acc_sizes_d(op.dimensions);
	std::vector<size_t> acc_sizes_s(op.dimensions);
	acc_sizes_d[op.dimensions - 1] = 1;
	acc_sizes_s[op.dimensions - 1] = 1;
	for (int dim = op.dimensions - 2; dim >= 0; dim--) {
		acc_sizes_d[dim] = acc_sizes_d[dim + 1] * op.shape[dim + 1];
		acc_sizes_s[dim] = acc_sizes_s[dim + 1] * pred.shape[dim + 1];
	}
	// to get the index in the source array we first calculate the
	// indices and reproject
	index_defs += "{\nint working_index = index;\nindex = 0;\n";
	for (int dim = 0; dim < op.dimensions; dim++) {
		index_defs += "index += ((working_index /" +
					  to_string(acc_sizes_d[dim]) + ") % " +
					  to_string(op.shape[dim]) + ") * " +
					  to_string(acc_sizes_s[transposition[dim]]) + ";\n";
		index_defs += "working_index %= " + to_string(acc_sizes_d[dim]) + ";\n";
	}
	index_defs += "}\n";
	compiler_state.index_defs = index_defs;
	compiler_state.code.prepend("index = old_index" + to_string(old_idx) +
								";\n");
	compiler_state.code.prepend(
		"const " + typeString(node->operation.data_type) + " " + name + " = v" +
		to_string(compiler_state.variable_index + 1) + ";\n");
	return 0;
}
void TransposeImpl::execute_cpu(const FGraphNode *node,
								vector<CPUResultData> predecessor_data,
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
int ConcatImpl::generate_ocl_lazy(const FGraphNode *node, string name,
								  OCLLazyCodegenState &compiler_state) {
	FGraphNode *a = node->predecessors[0];
	FGraphNode *b = node->predecessors[1];
	const unsigned int old_idx = compiler_state.num_indices++;
	const unsigned int ax =
		((unsigned int *)node->operation.additional_data)[0];
	size_t acc_size_last = 1;
	for (int i = node->operation.dimensions - 2; i >= (int)ax; i--)
		acc_size_last *= node->operation.shape[i + 1];
	Twine index_defs;
	index_defs += "long old_index" + to_string(old_idx) + " = index;\n";
	const string sx = "index / " + to_string(acc_size_last);
	const string sc =
		ax > 0 ? "(" + sx + ") % " + to_string(node->operation.shape[ax]) : sx;
	const string if_em = sc + " < " + to_string(a->operation.shape[ax]);
	index_defs += "index = " + if_em +
				  " ? "
				  "(" +
				  sx + " / " + to_string(node->operation.shape[ax]) + ") * " +
				  to_string(acc_size_last * a->operation.shape[ax]) + " + (" +
				  sc + ") * " + to_string(acc_size_last) + " + (index % " +
				  to_string(acc_size_last) + "): (" + sx + " / " +
				  to_string(node->operation.shape[ax]) + ") * " +
				  to_string(acc_size_last * b->operation.shape[ax]) + " + ((" +
				  sc + ") - " + to_string(a->operation.shape[ax]) + ") * " +
				  to_string(acc_size_last) + " + (index % " +
				  to_string(acc_size_last) + ");\n";
	compiler_state.index_defs = index_defs;
	compiler_state.code.prepend(
		"index = old_index" + to_string(old_idx) + ";\n" +
		typeString(node->operation.data_type) + " " + name + " = " + if_em +
		" ? v" + to_string(compiler_state.variable_index + 1) + " : v" +
		to_string(compiler_state.variable_index + 2) + ";\n");
	return 0;
}
void ConcatImpl::execute_cpu(const FGraphNode *node,
							 vector<CPUResultData> predecessor_data,
							 void *__restrict__ result, size_t from,
							 size_t size) {
	BINARY_EXECUTE_MONOTON_IMPL
}
