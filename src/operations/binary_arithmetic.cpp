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
#include "binary_arithmetic.hpp"
#include "../utils.hpp"
#include "flint.h"

using namespace std;
FGraphNode *AddImpl::local_gradient(FGraphNode *y, int dx_i,
									FGraphNode *prev_adj) {
	return (dx_i == 0 || dx_i == 1) ? prev_adj : nullptr;
}
template <typename T, typename A, typename B>
void AddImpl::binary_expression(T *__restrict__ result,
								const A *__restrict__ data1,
								const B *__restrict__ data2, size_t from,
								size_t size, size_t index_man_1,
								size_t inv_man_1, size_t index_man_2,
								size_t inv_man_2, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++) {
		result[i] = data1[(i / inv_man_1) % index_man_1] +
					data2[(i / inv_man_2) % index_man_2];
	}
}
int AddImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
							   OCLLazyCodegenState &compiler_state) {
	string type = type_string(node->operation.data_type);
	compiler_state.code.prepend(
		"const " + type + " " + name + " = v" +
		to_string(compiler_state.variable_index + 1) + " + v" +
		to_string(compiler_state.variable_index + 2) + ";\n");
	return OCL_LAZY_INVERSE_BROADCASTING;
}
std::vector<bool> AddImpl::reuse_parameter_binary_impl(const FGraphNode *node) {
	const FOperation op = node->operation;
	std::vector<bool> result(node->num_predecessor, false);
	for (int i = 0; i < node->num_predecessor; i++) {
		const FOperation pred = node->predecessors[i]->operation;
		if (op.dimensions == pred.dimensions &&
			type_size(op.data_type) == type_size(pred.data_type)) {
			result[i] = true;
			for (int j = 0; j < pred.dimensions; j++) {
				if (op.shape[j] != pred.shape[j]) {
					result[i] = false;
					break;
				}
			}
		}
	}
	return result;
}
FGraphNode *SubImpl::local_gradient(FGraphNode *y, int dx_i,
									FGraphNode *prev_adj) {
	if (dx_i == 0)
		return prev_adj;
	else if (dx_i == 1)
		return fneg(prev_adj);
	else
		return nullptr;
}
template <typename T, typename A, typename B>
void SubImpl::binary_expression(T *__restrict__ result,
								const A *__restrict__ data1,
								const B *__restrict__ data2, size_t from,
								size_t size, size_t index_man_1,
								size_t inv_man_1, size_t index_man_2,
								size_t inv_man_2, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++) {
		result[i] = data1[(i / inv_man_1) % index_man_1] -
					data2[(i / inv_man_2) % index_man_2];
	}
}
int SubImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
							   OCLLazyCodegenState &compiler_state) {
	string type = type_string(node->operation.data_type);
	compiler_state.code.prepend(
		"const " + type + " " + name + " = v" +
		to_string(compiler_state.variable_index + 1) + " - v" +
		to_string(compiler_state.variable_index + 2) + ";\n");
	return OCL_LAZY_INVERSE_BROADCASTING;
}
FGraphNode *MulImpl::local_gradient(FGraphNode *y, int dx_i,
									FGraphNode *prev_adj) {
	if (0 == dx_i) {
		return fmul(prev_adj, y->predecessors[1]);
	} else if (1 == dx_i) {
		return fmul(prev_adj, y->predecessors[0]);
	} else
		return nullptr;
}
template <typename T, typename A, typename B>
void MulImpl::binary_expression(T *__restrict__ result,
								const A *__restrict__ data1,
								const B *__restrict__ data2, size_t from,
								size_t size, size_t index_man_1,
								size_t inv_man_1, size_t index_man_2,
								size_t inv_man_2, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++) {
		result[i] = data1[(i / inv_man_1) % index_man_1] *
					data2[(i / inv_man_2) % index_man_2];
	}
}
int MulImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
							   OCLLazyCodegenState &compiler_state) {
	string type = type_string(node->operation.data_type);
	compiler_state.code.prepend(
		"const " + type + " " + name + " = v" +
		to_string(compiler_state.variable_index + 1) + " * v" +
		to_string(compiler_state.variable_index + 2) + ";\n");
	return OCL_LAZY_INVERSE_BROADCASTING;
}
template <typename T, typename A, typename B>
void DivImpl::binary_expression(T *__restrict__ result,
								const A *__restrict__ data1,
								const B *__restrict__ data2, size_t from,
								size_t size, size_t index_man_1,
								size_t inv_man_1, size_t index_man_2,
								size_t inv_man_2, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++) {
		result[i] = data1[(i / inv_man_1) % index_man_1] /
					data2[(i / inv_man_2) % index_man_2];
	}
}
int DivImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
							   OCLLazyCodegenState &compiler_state) {
	string type = type_string(node->operation.data_type);
	compiler_state.code.prepend(
		"const " + type + " " + name + " = v" +
		to_string(compiler_state.variable_index + 1) + " / v" +
		to_string(compiler_state.variable_index + 2) + ";\n");
	return OCL_LAZY_INVERSE_BROADCASTING;
}
FGraphNode *DivImpl::local_gradient(FGraphNode *y, int dx_i,
									FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	FGraphNode *b = y->predecessors[1];
	if (0 == dx_i) {
		// d(a / b)/da = d(a * b^(-1))/da = b^(-1)
		return fdiv(prev_adj, b);
	} else if (1 == dx_i) {
		// d(a / b)/db = d(a * b^(-1))/db = -a * b^(-2)
		return fneg(fdiv(fmul(prev_adj, a), fpow(b, 2.f)));
	} else
		return nullptr;
}

FGraphNode *PowImpl::local_gradient(FGraphNode *y, int dx_i,
									FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	FGraphNode *b = y->predecessors[1];
	if (0 == dx_i) {
		// x^b / dx = b*x^(b-1)
		return fmul(prev_adj, fmul(b, fpow(a, fsub(b, 1))));
	} else if (1 == dx_i) {
		// a^x / dx = a^x * ln(a)
		// has to be zero when a < 0 since not differentiable
		return fmul(prev_adj, fmul(fmul(fadd(fsign(a), 1), 0.5f),
								   fmul(fpow(a, b), flog(fabs_g(a)))));
	} else
		return nullptr;
}
template <typename T, typename A, typename B>
void PowImpl::binary_expression(T *__restrict__ result,
								const A *__restrict__ data1,
								const B *__restrict__ data2, size_t from,
								size_t size, size_t index_man_1,
								size_t inv_man_1, size_t index_man_2,
								size_t inv_man_2, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++) {
		result[i] = pow(data1[(i / inv_man_1) % index_man_1],
						data2[(i / inv_man_2) % index_man_2]);
	}
}
int PowImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
							   OCLLazyCodegenState &compiler_state) {
	const string type = type_string(node->operation.data_type);
	Twine &code = compiler_state.code;
	const int variable_index = compiler_state.variable_index;
	const FOperation x = node->predecessors[0]->operation;
	const FOperation y = node->predecessors[1]->operation;
	if ((x.data_type == F_FLOAT32 || x.data_type == F_FLOAT64) &&
		(y.data_type == F_FLOAT32 || y.data_type == F_FLOAT64))
		code.prepend("const " + type + " " + name + " = pow((" + type + ")v" +
					 to_string(variable_index + 1) + ", (" + type + ")v" +
					 to_string(variable_index + 2) + ");\n");
	else if (x.data_type == F_INT64 &&
			 (y.data_type == F_INT32 || y.data_type == F_INT64))
		code.prepend("const " + type + " " + name + " = (long)pown((double)v" +
					 to_string(variable_index + 1) + ", (int)v" +
					 to_string(variable_index + 2) + ");\n");
	else if (x.data_type == F_INT32 &&
			 (y.data_type == F_INT32 || y.data_type == F_INT64))
		code.prepend("const " + type + " " + name + " = (int)pown((float)v" +
					 to_string(variable_index + 1) + ", (int)v" +
					 to_string(variable_index + 2) + ");\n");
	else
		code.prepend("const " + type + " " + name + " = pow((double)v" +
					 to_string(variable_index + 1) + ", (double)v" +
					 to_string(variable_index + 2) + ");\n");
	return OCL_LAZY_INVERSE_BROADCASTING;
}
void SubImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	BINARY_EXECUTE_IMPL
}
void AddImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	BINARY_EXECUTE_IMPL
}
void MulImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	BINARY_EXECUTE_IMPL
}
void DivImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	BINARY_EXECUTE_IMPL
}
void PowImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	BINARY_EXECUTE_IMPL
}
