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
#include "../utils.hpp"
#include "flint.h"

using namespace std;

FGraphNode *SliceImpl::local_gradient(FGraphNode *y, int dx_i,
									  FGraphNode *prev_adj) {
	const FGraphNode *a = y->predecessors[0];
	const FSlice *slice = (FSlice *)y->operation.additional_data;
	std::vector<size_t> start(a->operation.dimensions);
	for (int i = 0; i < a->operation.dimensions; i++) {
		start[i] = slice->step[i] >= 0 ? slice->start[i] : slice->end[i] + 1;
	}
	return fextend_step(prev_adj, a->operation.shape, start.data(),
						slice->step);
}
template <typename T>
void SliceImpl::unary_expression(T *__restrict__ result,
								 const T *__restrict__ data, size_t from,
								 size_t size, const FGraphNode *curr) {
	FOperation pred = curr->predecessors[0]->operation;
	FSlice *slice = (FSlice *)curr->operation.additional_data;
	std::vector<size_t> acc_sizes = calc_acc_sizes(curr->operation);
	std::vector<size_t> acc_sizes_pred =
		calc_acc_sizes(pred.dimensions, pred.shape);
	// calculate start and step size in flattened array
	size_t start = 0;
	for (unsigned int d = 0; d < curr->operation.dimensions; d++) {
		start += slice->start[d] * acc_sizes_pred[d];
	}
	// calculate for each entry corresponding element
	for (size_t i = from; i < from + size; i++) {
		if (curr->operation.op_type == FGEN_CONSTANT)
			result[i] = data[0];
		else {
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
}
int SliceImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
								 OCLLazyCodegenState &compiler_state) {
	FSlice *slice = (FSlice *)node->operation.additional_data;
	const string type = type_string(node->operation.data_type);
	// the offset moves when a graph is reused on different slices, as an
	// argument it keeps the code cacheable
	for (int d = 0; d < node->operation.dimensions; d++) {
		if (slice->start[d] != 0) {
			const auto &exprs =
				compiler_state.pred_index_maps[0].expr_per_dim[d].exprs;
			compiler_state.declare_atom(
				exprs[exprs.size() - 1].second,
				compiler_state.addScalar(F_INT64, slice->start[d]));
		}
	}
	compiler_state.code.prepend("const " + type + " " + name + " = v" +
								to_string(compiler_state.variable_index + 1) +
								";\n");
	return 0;
}
void SliceImpl::execute_cpu(const FGraphNode *node,
							std::vector<CPUResultData> predecessor_data,
							void *__restrict__ result, size_t from,
							size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
bool SliceImpl::mutate_index(const FGraphNode *node,
							 OCLLazyCodegenState &state) {
	// we have C[i] = P[start + i * step] per dimension

	FOperation pred = node->predecessors[0]->operation;
	FSlice *slice = (FSlice *)node->operation.additional_data;
	IndexMap &pred_map = state.pred_index_maps.emplace_back();
	pred_map.current_depth = state.index_map.current_depth;
	pred_map.expr_per_dim.resize(state.index_map.expr_per_dim.size());
	for (int i = 0; i < node->operation.dimensions; i++) {
		IndexExpr &pred_expr = pred_map.expr_per_dim[i];
		IndexExpr &curr_expr = state.index_map.expr_per_dim[i];
		pred_expr.a = curr_expr.a;
		// * step has to be added to each term
		pred_expr.exprs = curr_expr.exprs;
		for (int j = 0; j < pred_expr.exprs.size(); j++) {
			pred_expr.exprs[j].first *= slice->step[i];
		}
		// add the start
		if (slice->start[i] != 0) {
			int atom = state.add_atom("slice_start" +
										  to_string(state.index_atoms.size()),
									  slice->start[i], slice->start[i],
									  state.index_map.current_depth);
			pred_expr.exprs.emplace_back(1, atom);
		}
		pred_expr.derive_bound(state.index_atoms);
	}
	return true;
}

FGraphNode *ExtendImpl::local_gradient(FGraphNode *y, int dx_i,
									   FGraphNode *prev_adj) {
	const FGraphNode *a = y->predecessors[0];
	const FExtend *extend = (FExtend *)y->operation.additional_data;
	std::vector<long> start(a->operation.dimensions);
	std::vector<long> ends(a->operation.dimensions);
	std::vector<long> steps(a->operation.dimensions);
	for (int i = 0; i < a->operation.dimensions; i++) {
		start[i] = extend->start[i];
		ends[i] = a->operation.shape[i] * extend->step[i] + extend->start[i];
		steps[i] = extend->step[i];
	}
	return fslice_step(prev_adj, start.data(), ends.data(), steps.data());
}
template <typename T>
void ExtendImpl::unary_expression(T *__restrict__ result,
								  const T *__restrict__ data, size_t from,
								  size_t size, const FGraphNode *curr) {
	FOperation pred = curr->predecessors[0]->operation;
	FExtend *extend = (FExtend *)curr->operation.additional_data;
	std::vector<size_t> acc_sizes = calc_acc_sizes(curr->operation);
	std::vector<size_t> acc_sizes_pred =
		calc_acc_sizes(pred.dimensions, pred.shape);
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
		if (pred.op_type == FGEN_CONSTANT)
			j = 0;
		result[i] = set_zero ? 0 : data[j];
	}
}
int ExtendImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
								  OCLLazyCodegenState &compiler_state) {
	const FOperation pred = node->predecessors[0]->operation;
	const string type = type_string(node->operation.data_type);
	const FExtend *extend = (FExtend *)node->operation.additional_data;
	std::string keep_cond;
	const auto keep_if = [&keep_cond](const std::string &cond) {
		if (!keep_cond.empty())
			keep_cond += " && ";
		keep_cond += cond;
	};
	for (long d = 0; d < node->operation.dimensions; d++) {
		const long start = (long)extend->start[d];
		const long step =
			extend->step[d] < 0 ? -(long)extend->step[d] : extend->step[d];
		const IndexExpr &coord_expr =
			compiler_state.index_map.expr_per_dim[d];
		const string coord = compiler_state.to_code(coord_expr);
		const string shifted =
			start == 0 ? coord
					   : "(" + coord + " - " + to_string(start) + ")";
		// the coordinate's own bound decides which of the three tests this
		// dimension still needs
		if (coord_expr.bound.first < start)
			keep_if(coord + " >= " + to_string(start));
		if (step != 1)
			keep_if(shifted + " % " + to_string(step) + " == 0");
		const long behind = start + step * (long)pred.shape[d];
		if (coord_expr.bound.second >= behind)
			keep_if(coord + " < " + to_string(behind));
		// the coordinate of the parameter is the one of the extension divided
		// by the step, which is only read where the condition says it is exact
		if (step != 1) {
			const auto &exprs =
				compiler_state.pred_index_maps[0].expr_per_dim[d].exprs;
			compiler_state.declare_atom(exprs[exprs.size() - 1].second,
										shifted + " / " + to_string(step));
		}
	}
	const string par = "v" + to_string(compiler_state.variable_index + 1);
	if (keep_cond.empty()) {
		// when nothing is padded away the extension only remaps
		compiler_state.code.prepend(type + " " + name + " = " + par + ";\n");
		return 0;
	}
	compiler_state.pred_wrappers = {
		{type + " " + name + " = 0;\nif(" + keep_cond + "){\n",
		 name + " = " + par + ";\n}\n"}};
	return 0;
}
bool ExtendImpl::mutate_index(const FGraphNode *node,
							  OCLLazyCodegenState &state) {
	const FOperation pred = node->predecessors[0]->operation;
	const FExtend *extend = (FExtend *)node->operation.additional_data;
	IndexMap pred_map;
	pred_map.current_depth = state.index_map.current_depth;
	pred_map.expr_per_dim.resize(node->operation.dimensions);
	for (int d = 0; d < node->operation.dimensions; d++) {
		const long step =
			extend->step[d] < 0 ? -(long)extend->step[d] : extend->step[d];
		IndexExpr &expr = pred_map.expr_per_dim[d];
		if (step == 1) {
			// the extension only shifts this dimension
			expr = state.index_map.expr_per_dim[d];
			expr.a -= (long)extend->start[d];
		} else {
			// only every `step`th entry comes from the parameter, which is a
			// division. `generate_ocl_lazy` emits it for this atom
			const int atom = state.add_atom(
				"extend_index_" + to_string(state.index_atoms.size()), 0,
				(long)pred.shape[d] - 1, state.index_map.current_depth);
			expr.a = 0;
			expr.exprs.push_back({1, atom});
		}
		if (extend->step[d] < 0) {
			// a negative step reads the dimension backwards
			for (auto &[c, atom] : expr.exprs)
				c = -c;
			expr.a = (long)pred.shape[d] - 1 - expr.a;
		}
		// everything outside of the parameter is guarded away, so the
		// coordinate stays inside of it even where the terms say otherwise
		expr.bound = {0, (long)pred.shape[d] - 1};
	}
	state.pred_index_maps.push_back(pred_map);
	return true;
}
void ExtendImpl::execute_cpu(const FGraphNode *node,
							 std::vector<CPUResultData> predecessor_data,
							 void *__restrict__ result, size_t from,
							 size_t size){UNARY_EXECUTE_MONOTON_IMPL}

FGraphNode *IndexImpl::local_gradient(FGraphNode *y, int dx_i,
									  FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	FGraphNode *b = y->predecessors[1];
	if (0 == dx_i) {
		FGraphNode *g =
			constant_tensor(0, a->operation.data_type, a->operation.shape,
							a->operation.dimensions);
		return findex_set(g, prev_adj, b);
	} else
		return constant_tensor(0, b->operation.data_type, b->operation.shape,
							   b->operation.dimensions);
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
		if (a->operation.op_type == FGEN_CONSTANT)
			result[i] = data1[0];
		else {
			const size_t base = i / (acc_sizes_ax * op.shape[axis]);
			const size_t rest = i % acc_sizes_ax;
			const size_t ind =
				(size_t)data2[b->operation.op_type == FGEN_CONSTANT
								  ? 0
								  : (i / acc_sizes_ax)];
			result[i] = data1[(base * acc_sizes_ax * a->operation.shape[axis]) +
							  (ind * acc_sizes_ax) + rest];
		}
	}
}
int IndexImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
								 OCLLazyCodegenState &compiler_state) {
	FGraphNode *a = node->predecessors[0];
	FGraphNode *b = node->predecessors[1];
	const unsigned int axis = b->operation.dimensions - 1;
	const string type = type_string(node->operation.data_type);
	const string par1 = "v" + to_string(++compiler_state.variable_index);
	const string par2 = "v" + to_string(++compiler_state.variable_index);
	compiler_state.code.prepend(type + " " + name + " = " + par1 + ";\n");
	const IndexMap coords = compiler_state.index_map;
	// `b` holds one index per element in front of and including the axis
	IndexMap bmap;
	bmap.current_depth = coords.current_depth;
	bmap.expr_per_dim.assign(coords.expr_per_dim.begin(),
							 coords.expr_per_dim.begin() + axis + 1);
	compiler_state.todo.push_front({b, par2, false, bmap});
	// `a` is read at the same coordinates, except on the axis, where the
	// entry of `b` says which one to take
	IndexMap amap = coords;
	IndexExpr &on_axis = amap.expr_per_dim[axis];
	on_axis.a = 0;
	on_axis.exprs = {{1, compiler_state.add_atom(
							 par2, 0, (long)a->operation.shape[axis] - 1,
							 coords.current_depth, true)}};
	on_axis.derive_bound(compiler_state.index_atoms);
	compiler_state.todo.push_front({a, par1, false, amap});
	return OCL_LAZY_DONT_PUSH_PREDS;
}
void IndexImpl::execute_cpu(const FGraphNode *node,
							std::vector<CPUResultData> predecessor_data,
							void *__restrict__ result, size_t from,
							size_t size){BINARY_EXECUTE_IMPL}

FGraphNode *SetIndexImpl::local_gradient(FGraphNode *y, int dx_i,
										 FGraphNode *prev_adj) {
	FGraphNode *b = y->predecessors[1];
	FGraphNode *i = y->predecessors[2];
	// a[i] = b
	if (0 == dx_i) {
		FGraphNode *g =
			constant_tensor(0, b->operation.data_type, b->operation.shape,
							b->operation.dimensions);
		// remove values that have been overwritten
		return findex_set(prev_adj, g, i);
	} else {
		// filter for b relevant elements
		return findex(prev_adj, i);
	}
}
template <typename T>
void SetIndexImpl::execute_cpu_typed(
	const FGraphNode *node, std::vector<CPUResultData> predecessor_data,
	T *__restrict__ result, size_t from, size_t size) {
	// TODO Improve (somehow)
	const CPUResultData a = predecessor_data[0];
	const CPUResultData b = predecessor_data[1];
	const CPUResultData c = predecessor_data[2];
	const bool ac = node->predecessors[0]->operation.op_type == FGEN_CONSTANT;
	const bool bc = node->predecessors[1]->operation.op_type == FGEN_CONSTANT;
	const bool cc = node->predecessors[2]->operation.op_type == FGEN_CONSTANT;
	const unsigned int axis = c.shape.size() - 1;
	const FOperation op = node->operation;
	// get index of result, index tensor, reproject index
	size_t acc_sizes_ax = 1;
	for (int i = axis + 1; i < op.dimensions; i++)
		acc_sizes_ax *= op.shape[i];

	for (size_t i = from; i < from + size; i++) {
		const size_t base = i / (acc_sizes_ax * op.shape[axis]);
		const size_t rest = i % acc_sizes_ax;
		const long axi = (i / acc_sizes_ax) % op.shape[axis];
		const size_t base_ind = base * c.shape[axis];
		bool found_something = false;
		T val = 0;
		// iterate over last dimension and find all correct indices
		for (size_t j = base_ind; j < base_ind + c.shape[axis]; j++) {
			const long ind =
				(long)(c.type == F_INT32 ? ((int *)c.data)[cc ? 0 : j]
										 : ((long *)c.data)[cc ? 0 : j]);
			if (ind == axi) {
				found_something = true;
				val += ((T *)b.data)[bc ? 0 : (j * acc_sizes_ax + rest)];
			}
		}
		// if at least one index was found -> only sum of elements of b.
		// `result` may be the array of `a`, so it is written exactly once
		result[i] = found_something ? val : ((T *)a.data)[ac ? 0 : i];
	}
}
int SetIndexImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
									OCLLazyCodegenState &compiler_state) {
	FGraphNode *a = node->predecessors[0];
	FGraphNode *b = node->predecessors[1];
	FGraphNode *c = node->predecessors[2];
	const FOperation op = node->operation;
	const unsigned int axis = c->operation.dimensions - 1;
	const string par2 = compiler_state.findOrInsertParameter(b),
				 par3 = compiler_state.findOrInsertParameter(c);
	const bool bc = b->operation.op_type == FGEN_CONSTANT;
	const bool cc = c->operation.op_type == FGEN_CONSTANT;
	// a may be calculated lazily
	const string par1 = "v" + to_string(++compiler_state.variable_index);
	size_t acc_sizes_ax = 1;
	for (int i = axis + 1; i < op.dimensions; i++)
		acc_sizes_ax *= op.shape[i];
	// the coordinates in front of the axis choose the indices to compare
	// against, the one of the axis is what they are compared to and the ones
	// behind it address the entry inside the window
	const IndexMap coords = compiler_state.index_map;
	IndexMap in_front, behind;
	in_front.expr_per_dim.assign(coords.expr_per_dim.begin(),
								 coords.expr_per_dim.begin() + axis);
	behind.expr_per_dim.assign(coords.expr_per_dim.begin() + axis + 1,
							   coords.expr_per_dim.end());
	const std::string base =
		axis > 0 ? compiler_state.to_code(
					   in_front.flatten(op.shape, axis))
				 : "0";
	const std::string rest =
		behind.expr_per_dim.empty()
			? "0"
			: compiler_state.to_code(behind.flatten(
				  op.shape + axis + 1, op.dimensions - axis - 1));
	const std::string axi = compiler_state.to_code(coords.expr_per_dim[axis]);
	const std::string base_ind =
		"(" + base + ") * " + to_string(c->operation.shape[axis]);
	const string type = type_string(node->operation.data_type);
	compiler_state.code.prepend(
		type + " " + name +
		" = 0;\n"
		"{const long base_ind = " +
		base_ind +
		";\n"
		" const long axi = " +
		axi +
		";\n"
		" const long rest = " +
		rest +
		";\n"
		"int found_something = false;\n"
		" for(long j = 0; j < " +
		to_string(c->operation.shape[axis]) +
		"; j++){\n"
		"  const long ind = " +
		par3 + (cc ? "[0]" : "[base_ind + j];\n") + "  if(ind == axi) {\n   " +
		name + " += " + par2 +
		(bc ? "[0]"
			: "[(base_ind + j) * " + to_string(acc_sizes_ax) + " + rest]") +
		";\n"
		"   found_something = true;\n"
		"  }\n"
		" }\n"
		" if(!found_something) " +
		name + " = " + par1 +
		";\n"
		"}\n");
	compiler_state.todo.push_front({a, par1, false, coords});
	return OCL_LAZY_DONT_PUSH_PREDS;
}
void SetIndexImpl::execute_cpu(const FGraphNode *node,
							   std::vector<CPUResultData> predecessor_data,
							   void *__restrict__ result, size_t from,
							   size_t size) {
	EXECUTE_TYPED_IMPL
}
