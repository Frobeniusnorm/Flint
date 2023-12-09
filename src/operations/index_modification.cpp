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

using namespace std;

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
int SliceImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
								 OCLLazyCodegenState &compiler_state) {
	FOperation pred = node->predecessors[0]->operation;
	FSlice *slice = (FSlice *)node->operation.additional_data;
	unsigned int old_idx = compiler_state.num_indices++;
	const string type = typeString(node->operation.data_type);
	Twine index_defs = "int old_index" + to_string(old_idx) + " = index;\n";
	// flattened shape data
	std::vector<size_t> acc_sizes(node->operation.dimensions);
	std::vector<size_t> acc_sizes_pred(acc_sizes.size());
	for (long d = node->operation.dimensions - 1; d >= 0; d--) {
		if (d == node->operation.dimensions - 1) {
			acc_sizes[d] = 1;
			acc_sizes_pred[d] = 1;
		} else {
			acc_sizes_pred[d] = acc_sizes_pred[d + 1] * pred.shape[d + 1];
			acc_sizes[d] = acc_sizes[d + 1] * node->operation.shape[d + 1];
		}
	}
	// calculate start
	size_t start = 0;
	std::vector<long> step(node->operation.dimensions);
	for (long d = 0; d < step.size(); d++) {
		start += slice->start[d] * acc_sizes_pred[d];
	}
	index_defs += "index = (" + to_string(start);
	// accumulate index
	for (long d = 0; d < node->operation.dimensions; d++) {
		index_defs +=
			" + (((" +
			(d == 0 ? string("index")
					: string("index %" + to_string(acc_sizes[d - 1]))) +
			") / " + to_string(acc_sizes[d]) + ") % " +
			to_string(node->operation.shape[d]) + ") * " +
			to_string(slice->step[d] * (long)acc_sizes_pred[d]);
	}
	index_defs += ") ;\n";
	compiler_state.code.prepend("index = old_index" + to_string(old_idx) +
								";\n");
	compiler_state.code.prepend("const " + type + " " + name + " = v" +
								to_string(compiler_state.variable_index + 1) +
								";\n");
	compiler_state.index_defs = index_defs;
	return 0;
}
std::string SliceImpl::generate_ocl_eager(FType res_type,
										  std::vector<FType> parameter_types) {}
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
int ExtendImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
								  OCLLazyCodegenState &compiler_state) {
	const FOperation pred = node->predecessors[0]->operation;
	const string type = typeString(node->operation.data_type);
	const FExtend *extend = (FExtend *)node->operation.additional_data;
	const unsigned int old_idx = compiler_state.num_indices++;
	Twine index_defs;
	index_defs += "int old_index" + to_string(old_idx) + " = index;\n";
	// flattened shape data
	std::vector<size_t> acc_sizes(node->operation.dimensions);
	std::vector<size_t> acc_sizes_pred(acc_sizes.size());
	for (long d = node->operation.dimensions - 1; d >= 0; d--) {
		if (d == node->operation.dimensions - 1) {
			acc_sizes[d] = 1;
			acc_sizes_pred[d] = 1;
		} else {
			acc_sizes_pred[d] = acc_sizes_pred[d + 1] * pred.shape[d + 1];
			acc_sizes[d] = acc_sizes[d + 1] * node->operation.shape[d + 1];
		}
	}
	// calculate start
	index_defs += "index = 0";
	std::string set_zero_cond = "if(";
	// accumulate index
	for (long d = 0; d < node->operation.dimensions; d++) {
		long step = extend->step[d];
		bool inv = step < 0;
		if (inv)
			step = -step;
		std::string dim_idx =
			"((" +
			(d == 0 ? string("index")
					: string("index %" + to_string(acc_sizes[d - 1]))) +
			") / " + to_string(acc_sizes[d]) + " - " +
			to_string(extend->start[d]) + ") / " + to_string(step);
		if (d != 0)
			set_zero_cond += " || ";
		// if di < start
		set_zero_cond +=
			"(" +
			(d == 0 ? string("index")
					: string("index %" + to_string(acc_sizes[d - 1]))) +
			") / " + to_string(acc_sizes[d]) + " < " +
			to_string(extend->start[d]);
		// if di % step != 0
		set_zero_cond +=
			" || ((" +
			(d == 0 ? string("index")
					: string("index %" + to_string(acc_sizes[d - 1]))) +
			") / " + to_string(acc_sizes[d]) + " - " +
			to_string(extend->start[d]) + ") % " + to_string(step) + " != 0";
		// if di >= shape
		set_zero_cond += " || " + dim_idx + " >= " + to_string(pred.shape[d]);

		// finish index
		if (inv)
			dim_idx =
				"(" + to_string(pred.shape[d]) + " - " + dim_idx + " - 1)";
		index_defs += " + " + dim_idx + " * " + to_string(acc_sizes_pred[d]);
	}
	index_defs += ";\nif(index < 0) index = 0;\n";
	compiler_state.index_defs = index_defs;
	compiler_state.code.prepend(set_zero_cond + ") " + name + " = 0;\n");
	compiler_state.code.prepend("index = old_index" + to_string(old_idx) +
								";\n");
	compiler_state.code.prepend(type + " " + name + " = v" +
								to_string(compiler_state.variable_index + 1) +
								";\n");
	return 0;
}
std::string ExtendImpl::generate_ocl_eager(FType res_type,
										   std::vector<FType> parameter_types) {
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
int IndexImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
								 OCLLazyCodegenState &compiler_state) {
	FGraphNode *a = node->predecessors[0];
	FGraphNode *b = node->predecessors[1];
	const FOperation op = node->operation;
	const unsigned int axis = b->operation.dimensions - 1;
	const string type = typeString(node->operation.data_type);
	string par1, par2;
	par1 = "v" + to_string(++compiler_state.variable_index);
	par2 = "v" + to_string(++compiler_state.variable_index);
	size_t acc_sizes_ax = 1;
	for (int i = axis + 1; i < op.dimensions; i++)
		acc_sizes_ax *= op.shape[i];

	const std::string base =
		"index / " + to_string(acc_sizes_ax * op.shape[axis]);
	const std::string rest = "index % " + to_string(acc_sizes_ax);
	unsigned int old_idx1 = compiler_state.num_indices++;
	unsigned int old_idx2 = compiler_state.num_indices++;
	std::string local_index_def1 = "index = old_index" + to_string(old_idx2) +
								   ";\nlong old_index" + to_string(old_idx1) +
								   " = index;\n";
	local_index_def1 += "index = " + base + " * " +
						to_string(acc_sizes_ax * a->operation.shape[axis]) +
						" + " + par2 + " * " + to_string(acc_sizes_ax) +
						" + (" + rest + ");\n";
	compiler_state.code.prepend("index = old_index" + to_string(old_idx1) +
								";\n" + type + " " + name + " = " + par1 +
								";\n");
	std::string local_index_def2 = "long old_index" + to_string(old_idx2) +
								   " = index;\n"
								   "index /= " +
								   to_string(acc_sizes_ax) + ";\n";
	compiler_state.todo.push_front({nullptr, local_index_def2});
	compiler_state.todo.push_front({b, par2});
	compiler_state.todo.push_front({nullptr, local_index_def1});
	compiler_state.todo.push_front({a, par1});
	return OCL_LAZY_DONT_PUSH_PREDS;
}
std::string IndexImpl::generate_ocl_eager(FType res_type,
										  std::vector<FType> parameter_types) {}
void IndexImpl::execute_cpu(const FGraphNode *node,
							std::vector<CPUResultData> predecessor_data,
							void *__restrict__ result, size_t from,
							size_t size) {
	BINARY_EXECUTE_IMPL
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
int SetIndexImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
									OCLLazyCodegenState &compiler_state) {
	FGraphNode *a = node->predecessors[0];
	FGraphNode *b = node->predecessors[1];
	FGraphNode *c = node->predecessors[2];
	const FOperation op = node->operation;
	const unsigned int axis = c->operation.dimensions - 1;
	const string par2 = compiler_state.findOrInsertParameter(b),
				 par3 = compiler_state.findOrInsertParameter(c);
	// a may be calculated lazily
	const string par1 = "v" + to_string(++compiler_state.variable_index);
	size_t acc_sizes_ax = 1;
	for (int i = axis + 1; i < op.dimensions; i++)
		acc_sizes_ax *= op.shape[i];
	const std::string base =
		"index / " + to_string(acc_sizes_ax * op.shape[axis]);
	const std::string rest = "index % " + to_string(acc_sizes_ax);
	const std::string axi = "(index / " + to_string(acc_sizes_ax) + ")%" +
							to_string(op.shape[axis]);
	const std::string ind =
		"(long) " + par3 + "[index / " + to_string(acc_sizes_ax) + "]";
	const std::string base_ind =
		base + " * " + to_string(c->operation.shape[axis]);
	const string type = typeString(node->operation.data_type);
	compiler_state.code.prepend(type + " " + name +
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
								par3 +
								"[base_ind + j];\n"
								"  if(ind == axi) {\n   " +
								name + " += " + par2 + "[(base_ind + j) * " +
								to_string(acc_sizes_ax) +
								" + rest];\n"
								"   found_something = true;\n"
								"  }\n"
								" }\n"
								" if(!found_something) " +
								name + " = " + par1 +
								";\n"
								"}\n");
	compiler_state.todo.push_front({a, par1});
	return OCL_LAZY_DONT_PUSH_PREDS;
}
std::string
SetIndexImpl::generate_ocl_eager(FType res_type,
								 std::vector<FType> parameter_types) {}
void SetIndexImpl::execute_cpu(const FGraphNode *node,
							   std::vector<CPUResultData> predecessor_data,
							   void *__restrict__ result, size_t from,
							   size_t size) {
	EXECUTE_TYPED_IMPL
}
