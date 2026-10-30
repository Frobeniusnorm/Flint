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
#include "../utils.hpp"
#include "flint.h"
#include "src/operations/indices.hpp"
#include <cstring>

using namespace std;

FGraphNode *FlattenImpl::local_gradient(FGraphNode *y, int dx_i,
										FGraphNode *prev_adj) {
	FGraphNode *prev = y->predecessors[0];
	return freshape(prev_adj, prev->operation.shape,
					prev->operation.dimensions);
}
void FlattenImpl::execute_cpu(const FGraphNode *node,
							  vector<CPUResultData> predecessor_data,
							  void *__restrict__ result, size_t from,
							  size_t size) {
	const CPUResultData pred = predecessor_data[0];
	switch (predecessor_data[0].type) {
	case F_INT32:
	case F_FLOAT32:
		if (node->predecessors[0]->operation.op_type != FGEN_CONSTANT) {
			memcpy((int *)result + from, (int *)pred.data + from, 4 * size);
		} else {
			for (int i = from; i < from + size; i++)
				((int *)result)[i] = ((int *)pred.data)[0];
		}
		break;
	case F_INT64:
	case F_FLOAT64:
		if (node->predecessors[0]->operation.op_type != FGEN_CONSTANT) {
			memcpy((long *)result + from, (long *)pred.data + from, 8 * size);
		} else {
			for (int i = from; i < from + size; i++)
				((long *)result)[i] = ((long *)pred.data)[0];
		}
		break;
	}
}
int FlattenImpl::generate_ocl_lazy(const FGraphNode *node, string name,
								   OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = v" + to_string(compiler_state.variable_index + 1) + ";\n");
	return 0;
}
bool FlattenImpl::mutate_index(const FGraphNode *node,
							   OCLLazyCodegenState &state) {
	const FOperation pred = node->predecessors[0]->operation;
	IndexExpr curr = state.flatten(state.index_map, node->operation.shape,
								   node->operation.dimensions);
	IndexMap mapped = split_atoms(curr, pred.shape, pred.dimensions, state);
	mapped.current_depth = state.index_map.current_depth;
	state.pred_index_maps.push_back(mapped);
	return true;
}

FGraphNode *ConversionImpl::local_gradient(FGraphNode *y, int dx_i,
										   FGraphNode *prev_adj) {
	return prev_adj;
}
template <typename T, typename A>
void ConversionImpl::unary_expression(T *__restrict__ result,
									  const A *__restrict__ data1, size_t from,
									  size_t size, const FGraphNode *curr) {
	const FGraphNode *prev = curr->predecessors[0];
	size_t total_el_size = 1;
	if (prev->operation.op_type != FGEN_CONSTANT)
		for (int i = 0; i < prev->operation.dimensions; i++)
			total_el_size *= prev->operation.shape[i];
	for (size_t i = from; i < from + size; i++)
		result[i] = (T)data1[i % total_el_size];
}
int ConversionImpl::generate_ocl_lazy(const FGraphNode *node, string name,
									  OCLLazyCodegenState &compiler_state) {
	const string type = type_string(node->operation.data_type);
	compiler_state.code.prepend(
		"const " + type + " " + name + " = (" + type + ")v" +
		to_string(compiler_state.variable_index + 1) + ";\n");
	return 0;
}
void ConversionImpl::execute_cpu(const FGraphNode *node,
								 vector<CPUResultData> predecessor_data,
								 void *__restrict__ result, size_t from,
								 size_t size){UNARY_EXECUTE_IMPL}

FGraphNode *RepeatImpl::local_gradient(FGraphNode *y, int dx_i,
									   FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	FGraphNode *grad = prev_adj;
	std::vector<size_t> orig_shape(prev_adj->operation.shape,
								   prev_adj->operation.shape +
									   prev_adj->operation.dimensions);
	for (int i = 0; i < a->operation.dimensions; i++) {
		if (a->operation.shape[i] != y->operation.shape[i]) {
			// reduce repeated gradient into correct shape
			std::vector<size_t> new_shape(orig_shape.size() + 1);
			// add extra dimension for reducing
			if (i > 0)
				std::memcpy(new_shape.data(), grad->operation.shape,
							sizeof(size_t) * i);
			new_shape[i] = orig_shape[i] / a->operation.shape[i];
			new_shape[i + 1] = a->operation.shape[i];
			if (orig_shape.size() - i > 1)
				std::memcpy(new_shape.data() + i + 2,
							grad->operation.shape + i + 1,
							sizeof(size_t) * (orig_shape.size() - i - 1));
			// reduce along that axis
			grad = freduce_sum(
				freshape(grad, new_shape.data(), new_shape.size()), i);
		}
	}
	return grad;
}
template <typename T>
void RepeatImpl::unary_expression(T *__restrict__ result,
								  const T *__restrict__ data, size_t from,
								  size_t size, const FGraphNode *curr) {
	const FOperation op = curr->operation;
	const FOperation pred = curr->predecessors[0]->operation;
	size_t total_el_size = 1;
	if (pred.op_type != FGEN_CONSTANT)
		for (int i = 0; i < pred.dimensions; i++)
			total_el_size *= pred.shape[i];
	// calculate number of elements per dimension entry for destination and
	// source
	vector<size_t> acc_sizes_d = calc_acc_sizes(op);
	vector<size_t> acc_sizes_s = calc_acc_sizes(pred.dimensions, pred.shape);
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
		result[i] = data[src_index % total_el_size];
	}
}
int RepeatImpl::generate_ocl_lazy(const FGraphNode *node, string name,
								  OCLLazyCodegenState &compiler_state) {
	const FOperation op = node->operation;
	const FOperation pred = node->predecessors[0]->operation;
	for (int i = 0; i < op.dimensions; i++) {
		if (op.shape[i] != pred.shape[i] && pred.shape[i] != 1) {
			compiler_state.declare_atom(
				compiler_state.pred_index_maps[0]
					.expr_per_dim[i]
					.exprs[0]
					.second,
				"(" +
					compiler_state.to_code(
						compiler_state.index_map.expr_per_dim[i]) +
					") % " + to_string(pred.shape[i]));
		}
	}

	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = v" + to_string(compiler_state.variable_index + 1) + ";\n");
	return 0;
}
void RepeatImpl::execute_cpu(const FGraphNode *node,
							 vector<CPUResultData> predecessor_data,
							 void *__restrict__ result, size_t from,
							 size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
bool RepeatImpl::mutate_index(const FGraphNode *node,
							  OCLLazyCodegenState &state) {
	const FOperation op = node->operation;
	const FOperation pred = node->predecessors[0]->operation;
	IndexMap pred_map;
	pred_map.current_depth = state.index_map.current_depth;
	pred_map.expr_per_dim.reserve(state.index_map.expr_per_dim.size());
	for (int i = 0; i < op.dimensions; i++) {
		if (op.shape[i] == pred.shape[i]) { // same
			pred_map.expr_per_dim.push_back(state.index_map.expr_per_dim[i]);
		} else if (pred.shape[i] == 1) {
			// a repeated constant has always index 0
			IndexExpr zero;
			zero.a = 0;
			zero.exprs = {};
			zero.bound = {0, 0};
			pred_map.expr_per_dim.push_back(zero);
		} else {
			// new atom with modulo
			int rep_ax = state.add_atom(
				"rep_index_" + to_string(state.index_atoms.size()), 0,
				pred.shape[i] - 1, state.index_map.current_depth);
			IndexExpr dim_idx;
			dim_idx.a = 0;
			dim_idx.exprs.push_back({1, rep_ax});
			dim_idx.derive_bound(state.index_atoms);
			pred_map.expr_per_dim.push_back(dim_idx);
		}
	}
	state.pred_index_maps.push_back(pred_map);
	return true;
}

FGraphNode *TransposeImpl::local_gradient(FGraphNode *y, int dx_i,
										  FGraphNode *prev_adj) {
	// the gradient undoes the transposition
	const int *transp = ((int *)y->operation.additional_data);
	const int dims = y->operation.dimensions;
	int inverse[dims];
	for (int i = 0; i < dims; i++)
		inverse[transp[i]] = i;
	return ftranspose(prev_adj, inverse);
}
template <typename T>
void TransposeImpl::unary_expression(T *__restrict__ result,
									 const T *__restrict__ data, size_t from,
									 size_t size, const FGraphNode *curr) {
	const FOperation op = curr->operation;
	const int *transposition = (int *)op.additional_data;
	const FOperation pred = curr->predecessors[0]->operation;
	size_t total_el_size = 1;
	if (pred.op_type != FGEN_CONSTANT)
		for (int i = 0; i < pred.dimensions; i++)
			total_el_size *= pred.shape[i];
	// calculate number of elements per dimension entry for destination and
	// source
	vector<size_t> acc_sizes_d = calc_acc_sizes(op);
	vector<size_t> acc_sizes_s = calc_acc_sizes(pred.dimensions, pred.shape);
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
		result[i] = data[src_index % total_el_size];
	}
}
int TransposeImpl::generate_ocl_lazy(const FGraphNode *node, string name,
									 OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = v" + to_string(compiler_state.variable_index + 1) + ";\n");
	return 0;
}
void TransposeImpl::execute_cpu(const FGraphNode *node,
								vector<CPUResultData> predecessor_data,
								void *__restrict__ result, size_t from,
								size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}

bool TransposeImpl::mutate_index(const FGraphNode *node,
								 OCLLazyCodegenState &state) {
	// retranspose indices
	IndexMap pred = state.index_map;
	int *transpositions = static_cast<int *>(node->operation.additional_data);
	for (int i = 0; i < node->operation.dimensions; i++) {
		int j = transpositions[i];
		// swap i and j in pred
		pred.expr_per_dim[j] = state.index_map.expr_per_dim[i];
	}
	state.pred_index_maps = {pred};
	return true;
}

FGraphNode *ConcatImpl::local_gradient(FGraphNode *y, int dx_i,
									   FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	FGraphNode *b = y->predecessors[1];
	unsigned int ax = *((unsigned int *)y->operation.additional_data);
	if (0 == dx_i) {
		std::vector<long> start(a->operation.dimensions);
		std::vector<long> stop(a->operation.dimensions);
		for (int i = 0; i < a->operation.dimensions; i++) {
			start[i] = 0;
			stop[i] = a->operation.shape[i];
		}
		return fslice(prev_adj, start.data(), stop.data());
	} else if (1 == dx_i) {
		std::vector<long> start(b->operation.dimensions);
		std::vector<long> stop(b->operation.dimensions);
		for (int i = 0; i < b->operation.dimensions; i++) {
			start[i] = ax == i ? a->operation.shape[i] : 0;
			stop[i] = ax == i ? a->operation.shape[i] + b->operation.shape[i]
							  : b->operation.shape[i];
		}
		return fslice(prev_adj, start.data(), stop.data());
	} else
		return nullptr;
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
	size_t total_el_size0 = 1;
	if (a->operation.op_type != FGEN_CONSTANT)
		for (int i = 0; i < a->operation.dimensions; i++)
			total_el_size0 *= a->operation.shape[i];
	size_t total_el_size1 = 1;
	if (b->operation.op_type != FGEN_CONSTANT)
		for (int i = 0; i < b->operation.dimensions; i++)
			total_el_size1 *= b->operation.shape[i];
	for (size_t index = from; index < from + size; index++) {
		size_t sx = index / acc_size_last;
		size_t sc = ax > 0 ? sx % curr->operation.shape[ax] : sx;
		// reproject to one of the tensors
		if (sc < a->operation.shape[ax]) {
			size_t ai = (sx / curr->operation.shape[ax]) * acc_size_last *
							a->operation.shape[ax] +
						sc * acc_size_last + index % acc_size_last;
			result[index] = data1[ai % total_el_size0];
		} else {
			size_t bi = (sx / curr->operation.shape[ax]) * acc_size_last *
							b->operation.shape[ax] +
						(sc - a->operation.shape[ax]) * acc_size_last +
						index % acc_size_last;
			result[index] = data2[bi % total_el_size1];
		}
	}
}
int ConcatImpl::generate_ocl_lazy(const FGraphNode *node, string name,
								  OCLLazyCodegenState &compiler_state) {
	FGraphNode *a = node->predecessors[0];
	FGraphNode *b = node->predecessors[1];
	const unsigned int ax =
		((unsigned int *)node->operation.additional_data)[0];
	const string type = type_string(node->operation.data_type);
	const string par1 = "v" + to_string(compiler_state.variable_index + 1);
	const string par2 = "v" + to_string(compiler_state.variable_index + 2);
	// the coordinate on the concatenated axis decides which of the two
	// parameters is read. Only that one is calculated, so each parameter is
	// wrapped in its branch of the condition
	const string in_a =
		compiler_state.to_code(compiler_state.index_map.expr_per_dim[ax]) +
		" < " + to_string(a->operation.shape[ax]);
	compiler_state.pred_wrappers = {
		{type + " " + name + " = 0;\nif(" + in_a + "){\n",
		 name + " = " + par1 + ";\n} else {\n"},
		{"", name + " = " + par2 + ";\n}\n"}};
	return 0;
}
bool ConcatImpl::mutate_index(const FGraphNode *node,
							  OCLLazyCodegenState &state) {
	const unsigned int ax =
		((unsigned int *)node->operation.additional_data)[0];
	for (int i = 0; i < node->num_predecessor; i++) {
		IndexMap &map = state.pred_index_maps.emplace_back(state.index_map);
		map.current_depth = state.index_map.current_depth;
		IndexExpr &expr = map.expr_per_dim[ax];
		// the second parameter starts where the first one ends
		if (i == 1)
			expr.a -= (long)node->predecessors[0]->operation.shape[ax];
		// each parameter is only read in its branch of the condition, which is
		// where its coordinate lies inside of it
		expr.bound = {0, (long)node->predecessors[i]->operation.shape[ax] - 1};
	}
	return true;
}
void ConcatImpl::execute_cpu(const FGraphNode *node,
							 vector<CPUResultData> predecessor_data,
							 void *__restrict__ result, size_t from,
							 size_t size) {
	BINARY_EXECUTE_MONOTON_IMPL
}
