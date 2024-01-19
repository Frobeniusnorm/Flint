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
#include "comparison.hpp"
#include "../utils.hpp"
#include "flint.h"
#include <cstring>
#include <random>

#define MIN_VAL(x, y) (x < y ? x : y)
#define MAX_VAL(x, y) (x < y ? y : x)

using namespace std;

FGraphNode *MinImpl::local_gradient(FGraphNode *y, int dx_i,
									FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	FGraphNode *b = y->predecessors[1];
	if (0 == dx_i) {
		return fmul(prev_adj, fadd(fless(a, b), fequal(a, b)));
	} else if (1 == dx_i)
		return fmul(prev_adj, fgreater(a, b));
	else
		return nullptr;
}
template <typename T, typename A, typename B>
void MinImpl::binary_expression(T *__restrict__ result,
								const A *__restrict__ data1,
								const B *__restrict__ data2, size_t from,
								size_t size, size_t index_man_1,
								size_t inv_man_1, size_t index_man_2,
								size_t inv_man_2, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = MIN_VAL(data1[(i / inv_man_1) % index_man_1],
							data2[(i / inv_man_2) % index_man_2]);
}
void MinImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	BINARY_EXECUTE_IMPL
}
int MinImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
							   OCLLazyCodegenState &compiler_state) {
	const std::string type = type_string(node->operation.data_type);
	compiler_state.code.prepend(
		"const " + type + " " + name + " = min((" + type + ")v" +
		to_string(compiler_state.variable_index + 1) + ", (" + type + ")v" +
		to_string(compiler_state.variable_index + 2) + ");\n");
	return OCL_LAZY_INVERSE_BROADCASTING;
}
std::string MinImpl::generate_ocl_eager(FType res_type,
										std::vector<FType> parameter_types) {
	return "if(index >= num_entries0 && index >= num_entries1) return;\n" +
		   type_string(parameter_types[0]) +
		   " a = P0[(index/inv_broad0)%num_entries0];\n" +
		   type_string(parameter_types[1]) +
		   " b = P1[(index/inv_broad1)%num_entries1];\n" +
		   "R[index] = a < b ? a : b;";
}
FGraphNode *MaxImpl::local_gradient(FGraphNode *y, int dx_i,
									FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	FGraphNode *b = y->predecessors[1];
	if (0 == dx_i)
		return fmul(prev_adj, fadd(fgreater(a, b), fequal(a, b)));
	else if (1 == dx_i)
		return fmul(prev_adj, fadd(fless(a, b), fequal(a, b)));
	else
		return nullptr;
}
template <typename T, typename A, typename B>
void MaxImpl::binary_expression(T *__restrict__ result,
								const A *__restrict__ data1,
								const B *__restrict__ data2, size_t from,
								size_t size, size_t index_man_1,
								size_t inv_man_1, size_t index_man_2,
								size_t inv_man_2, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = MAX_VAL(data1[(i / inv_man_1) % index_man_1],
							data2[(i / inv_man_2) % index_man_2]);
}
int MaxImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
							   OCLLazyCodegenState &compiler_state) {
	const std::string type = type_string(node->operation.data_type);
	compiler_state.code.prepend(
		"const " + type + " " + name + " = max((" + type + ")v" +
		to_string(compiler_state.variable_index + 1) + ", (" + type + ")v" +
		to_string(compiler_state.variable_index + 2) + ");\n");
	return OCL_LAZY_INVERSE_BROADCASTING;
}
std::string MaxImpl::generate_ocl_eager(FType res_type,
										std::vector<FType> parameter_types) {
	return "if(index >= num_entries0 && index >= num_entries1) return;\n" +
		   type_string(parameter_types[0]) +
		   " a = P0[(index/inv_broad0)%num_entries0];\n" +
		   type_string(parameter_types[1]) +
		   " b = P1[(index/inv_broad1)%num_entries1];\n" +
		   "R[index] = a >= b ? a : b;";
}
void MaxImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from,
						  size_t size){BINARY_EXECUTE_IMPL}

FGraphNode *LessImpl::local_gradient(FGraphNode *y, int dx_i,
									 FGraphNode *prev_adj) {
	return constant_tensor(0.0, F_FLOAT64, y->operation.shape,
						   y->operation.dimensions);
}
template <typename A, typename B>
void LessImpl::binary_expression(int *__restrict__ result,
								 const A *__restrict__ data1,
								 const B *__restrict__ data2, size_t from,
								 size_t size, size_t index_man_1,
								 size_t inv_man_1, size_t index_man_2,
								 size_t inv_man_2, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = data1[(i / inv_man_1) % index_man_1] <
							data2[(i / inv_man_2) % index_man_2]
						? 1
						: 0;
}
int LessImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
								OCLLazyCodegenState &compiler_state) {
	const std::string type = type_string(node->operation.data_type);
	compiler_state.code.prepend(
		"const " + type + " " + name + " = v" +
		to_string(compiler_state.variable_index + 1) + " < v" +
		to_string(compiler_state.variable_index + 2) + " ? 1 : 0;\n");
	return OCL_LAZY_INVERSE_BROADCASTING;
}
std::string LessImpl::generate_ocl_eager(FType res_type,
										 std::vector<FType> parameter_types) {
	return "if(index >= num_entries0 && index >= num_entries1) return;\n" +
		   type_string(parameter_types[0]) +
		   " a = P0[(index/inv_broad0)%num_entries0];\n" +
		   type_string(parameter_types[1]) +
		   " b = P1[(index/inv_broad1)%num_entries1];\n" +
		   "R[index] = a < b ? 1 : 0;";
}
void LessImpl::execute_cpu(const FGraphNode *node,
						   std::vector<CPUResultData> predecessor_data,
						   void *__restrict__ result, size_t from,
						   size_t size){DISPATCH_BINARY_OPERATION(int)}

FGraphNode *GreaterImpl::local_gradient(FGraphNode *y, int dx_i,
										FGraphNode *prev_adj) {
	return constant_tensor(0.0, F_FLOAT64, y->operation.shape,
						   y->operation.dimensions);
}
template <typename A, typename B>
void GreaterImpl::binary_expression(int *__restrict__ result,
									const A *__restrict__ data1,
									const B *__restrict__ data2, size_t from,
									size_t size, size_t index_man_1,
									size_t inv_man_1, size_t index_man_2,
									size_t inv_man_2, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = data1[(i / inv_man_1) % index_man_1] >
							data2[(i / inv_man_2) % index_man_2]
						? 1
						: 0;
}
int GreaterImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
								   OCLLazyCodegenState &compiler_state) {
	const std::string type = type_string(node->operation.data_type);
	compiler_state.code.prepend(
		"const " + type + " " + name + " = v" +
		to_string(compiler_state.variable_index + 1) + " > v" +
		to_string(compiler_state.variable_index + 2) + " ? 1 : 0;\n");
	return OCL_LAZY_INVERSE_BROADCASTING;
}
std::string
GreaterImpl::generate_ocl_eager(FType res_type,
								std::vector<FType> parameter_types) {
	return "if(index >= num_entries0 && index >= num_entries1) return;\n" +
		   type_string(parameter_types[0]) +
		   " a = P0[(index/inv_broad0)%num_entries0];\n" +
		   type_string(parameter_types[1]) +
		   " b = P1[(index/inv_broad1)%num_entries1];\n" +
		   "R[index] = a > b ? 1 : 0;";
}
void GreaterImpl::execute_cpu(const FGraphNode *node,
							  std::vector<CPUResultData> predecessor_data,
							  void *__restrict__ result, size_t from,
							  size_t size){DISPATCH_BINARY_OPERATION(int)}

FGraphNode *EqualImpl::local_gradient(FGraphNode *y, int dx_i,
									  FGraphNode *prev_adj) {
	return constant_tensor(0.0, F_FLOAT64, y->operation.shape,
						   y->operation.dimensions);
}
template <typename A, typename B>
void EqualImpl::binary_expression(int *__restrict__ result,
								  const A *__restrict__ data1,
								  const B *__restrict__ data2, size_t from,
								  size_t size, size_t index_man_1,
								  size_t inv_man_1, size_t index_man_2,
								  size_t inv_man_2, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = data1[(i / inv_man_1) % index_man_1] ==
							data2[(i / inv_man_2) % index_man_2]
						? 1
						: 0;
}
int EqualImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
								 OCLLazyCodegenState &compiler_state) {
	const std::string type = type_string(node->operation.data_type);
	const FOperation x = node->predecessors[0]->operation;
	const FOperation y = node->predecessors[1]->operation;
	compiler_state.code.prepend(
		"const " + type + " " + name + " = v" +
		to_string(compiler_state.variable_index + 1) + " + " +
		epsilon_for_type(x.data_type) + " >= v" +
		to_string(compiler_state.variable_index + 2) +
		" && "
		"v" +
		to_string(compiler_state.variable_index + 1) + " <= v" +
		to_string(compiler_state.variable_index + 2) + " + " +
		epsilon_for_type(y.data_type) + "? 1 : 0;\n");
	return OCL_LAZY_INVERSE_BROADCASTING;
}
std::string EqualImpl::generate_ocl_eager(FType res_type,
										  std::vector<FType> parameter_types) {
	return "if(index >= num_entries0 && index >= num_entries1) return;\n" +
		   type_string(parameter_types[0]) +
		   " a = P0[(index/inv_broad0)%num_entries0];\n" +
		   type_string(parameter_types[1]) +
		   " b = P1[(index/inv_broad1)%num_entries1];\n" + "R[index] = a + " +
		   epsilon_for_type(parameter_types[0]) + " >= b && a <= b + " +
		   epsilon_for_type(parameter_types[1]) + " ? 1 : 0;";
}
void EqualImpl::execute_cpu(const FGraphNode *node,
							std::vector<CPUResultData> predecessor_data,
							void *__restrict__ result, size_t from,
							size_t size) {
	DISPATCH_BINARY_OPERATION(int)
}
template <typename T>
void DropoutImpl::unary_expression(T *__restrict__ result,
								   const T *__restrict__ data1, size_t from,
								   size_t size, const FGraphNode *curr) {

	const double seed = ((double *)curr->operation.additional_data)[0];
	const double prob = ((double *)curr->operation.additional_data)[1];
	std::minstd_rand0 g1(seed * 1000 + from);
	for (size_t i = from; i < from + size; i++) {
		result[i] = ((g1() % 100000000) / 100000000.0) > prob ? data1[i] : 0;
	}
}
void DropoutImpl::execute_cpu(const FGraphNode *node,
							  std::vector<CPUResultData> predecessor_data,
							  void *__restrict__ result, size_t from,
							  size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
int DropoutImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
								   OCLLazyCodegenState &compiler_state) {

	const string type = type_string(node->operation.data_type);
	const double seed = ((double *)node->operation.additional_data)[0];
	const double prob = ((double *)node->operation.additional_data)[1];
	compiler_state.code.prepend(
		type + " " + name + " = 0;\n{\n double _random = sin(index + " +
		std::to_string(seed) +
		") * 43758.5453123;\n _random = min(_random - floor(_random), "
		"0.99999);\n" +
		name +
		" = "
		"_random > " +
		to_string(prob) + "?v" + to_string(compiler_state.variable_index + 1) +
		" : 0;\n"
		"}\n");
	return 0;
}
std::string
DropoutImpl::generate_ocl_eager(FType res_type,
								std::vector<FType> parameter_types) {
	return "if(index >= num_entriesR) return;\n"
		   "const double v = sin(index + time) * 43758.5453123;\n"
		   "const double r = min(v - floor(v), 0.99999);\n"
		   "R[index] = r > prob ? P0[index] : 0;\n";
}
FGraphNode *DropoutImpl::local_gradient(FGraphNode *y, int dx_i,
										FGraphNode *prev_adj) {
	const double *orig_data = (double *)y->operation.additional_data;
	const bool was_eager = fIsEagerExecution();
	if (was_eager)
		fDisableEagerExecution();
	FGraphNode *grad = fdropout(prev_adj, orig_data[1]);
	((double *)grad->operation.additional_data)[0] = orig_data[0];
	if (was_eager) {
		fEnableEagerExecution();
		fExecuteGraph(grad);
	}
	return grad;
}
std::string
DropoutImpl::generate_ocl_parameters_eager(FType res_type,
										   std::vector<FType> parameter_types) {
	return ", const __global " + type_string(parameter_types[0]) +
		   "* P0, const long num_entries0, const double "
		   "time, const double prob";
}
void DropoutImpl::push_additional_kernel_parameters(
	FGraphNode *node, cl_kernel kernel, cl_context context, int &par_index,
	std::list<cl_mem> &to_free) {
	const double seed = ((double *)node->operation.additional_data)[0];
	const double prob = ((double *)node->operation.additional_data)[1];
	if (clSetKernelArg(kernel, par_index++, sizeof(double), (void *)&seed) !=
		CL_SUCCESS) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Could not load Argument to kernel!");
		return;
	}
	if (clSetKernelArg(kernel, par_index++, sizeof(double), (void *)&prob) !=
		CL_SUCCESS) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Could not load Argument to kernel!");
		return;
	}
}
