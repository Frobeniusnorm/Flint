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

using namespace std;

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
	string type = typeString(node->operation.data_type);
	compiler_state.code.prepend(
		"const " + type + " " + name + " = v" +
		to_string(compiler_state.variable_index + 1) + " + v" +
		to_string(compiler_state.variable_index + 2) + ";\n");
	return OCL_LAZY_INVERSE_BROADCASTING;
}
std::string AddImpl::generate_ocl_eager(FType res_type,
							  std::vector<FType> parameter_types) {
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
	string type = typeString(node->operation.data_type);
	compiler_state.code.prepend(
		"const " + type + " " + name + " = v" +
		to_string(compiler_state.variable_index + 1) + " - v" +
		to_string(compiler_state.variable_index + 2) + ";\n");
	return OCL_LAZY_INVERSE_BROADCASTING;
}
std::string SubImpl::generate_ocl_eager(FType res_type,
							  std::vector<FType> parameter_types) {
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
	string type = typeString(node->operation.data_type);
	compiler_state.code.prepend(
		"const " + type + " " + name + " = v" +
		to_string(compiler_state.variable_index + 1) + " * v" +
		to_string(compiler_state.variable_index + 2) + ";\n");
	return OCL_LAZY_INVERSE_BROADCASTING;
}
std::string MulImpl::generate_ocl_eager(FType res_type,
							  std::vector<FType> parameter_types) {
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
	string type = typeString(node->operation.data_type);
	compiler_state.code.prepend(
		"const " + type + " " + name + " = v" +
		to_string(compiler_state.variable_index + 1) + " / v" +
		to_string(compiler_state.variable_index + 2) + ";\n");
	return OCL_LAZY_INVERSE_BROADCASTING;
}
std::string PowImpl::generate_ocl_eager(FType res_type,
							  std::vector<FType> parameter_types) {
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
	const string type = typeString(node->operation.data_type);
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
template <typename T, typename A, typename B>
void MatMulImpl::binary_expression(T *__restrict__ result,
								   const A *__restrict__ data1,
								   const B *__restrict__ data2, size_t from,
								   size_t size, size_t index_man_1,
								   size_t inv_man_1, size_t index_man_2,
								   size_t inv_man_2, const FGraphNode *curr) {
	FGraphNode *gnp1 = curr->predecessors[0], *gnp2 = curr->predecessors[1];
	size_t l = gnp1->operation.shape[gnp1->operation.dimensions - 2];
	size_t m = gnp1->operation.shape[gnp1->operation.dimensions - 1];
	size_t n = gnp2->operation.shape[gnp2->operation.dimensions - 1];
	for (size_t index = from; index < from + size; index++) {
		result[index] = 0;
		// indices in node matrix
		size_t j = (index % (l * n)) / n;
		size_t k = (index % (l * n)) % n;

		// matrix number of predecessors
		size_t base_p1 = 0;
		if (gnp1->operation.dimensions > 2) {
			// get matrix number of index and then reproject
			base_p1 = (index / (l * n)) * (l * m);
		}
		size_t base_p2 = 0;
		if (gnp2->operation.dimensions > 2) {
			// get matrix number of index and then reproject
			base_p2 = (index / (l * n)) * (m * n);
		}
		for (size_t i = 0; i < m; i++) {
			result[index] +=
				data1[base_p1 + j * m + i] * data2[base_p2 + i * n + k];
		}
	}
}
int MatMulImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
								  OCLLazyCodegenState &compiler_state) {
	string type = typeString(node->operation.data_type);
	string par1, par2;
	auto parameters = compiler_state.parameters;
	FGraphNode *gnp1 = node->predecessors[0], *gnp2 = node->predecessors[1];
	Twine& code = compiler_state.code;
	// we ignore the value assignment of the parameters since we
	// have to access the arrays directly parameter 1
	if (compiler_state.assigned_params.find(gnp1) != compiler_state.assigned_params.end()) {
		par1 = compiler_state.assigned_params[gnp1];
	} else {
		par1 = "P" + to_string(compiler_state.assigned_params.size());
		compiler_state.assigned_params.insert({gnp1, par1});
		parameters->push_back({gnp1, par1});
	}
	// parameter 2
	if (compiler_state.assigned_params.find(gnp2) != compiler_state.assigned_params.end()) {
		par2 = compiler_state.assigned_params[gnp2];
	} else {
		par2 = "P" + to_string(compiler_state.assigned_params.size());
		compiler_state.assigned_params.insert({gnp2, par2});
		parameters->push_back({gnp2, par2});
	}
	size_t l = gnp1->operation.shape[gnp1->operation.dimensions - 2];
	size_t m = gnp1->operation.shape[gnp1->operation.dimensions - 1];
	size_t n = gnp2->operation.shape[gnp2->operation.dimensions - 1];
	// we need to compute $name
	// indices j and k of $name
	string j = "((index % " + to_string(l * n) + ")/" + to_string(n) + ")";
	string k = "((index % " + to_string(l * n) + ")%" + to_string(n) + ")";
	// base index of matrix start of p1 and p2
	string base_p1 = "";
	if (gnp1->operation.dimensions > 2) {
		// get matrix number of index and then reproject
		base_p1 = "(index / " + to_string(l * n) + ") * " + to_string(l * m);
	} else
		base_p1 = "0";
	string base_p2 = "";
	if (gnp2->operation.dimensions > 2) {
		// get matrix number of index and then reproject
		base_p2 = "(index / " + to_string(l * n) + ") * " + to_string(m * n);
	} else
		base_p2 = "0";
	code.prepend("for(int i = 0; i < " + to_string(m) +
		   "; i++){\n"
		   "  " +
		   name + " += " + par1 + "[" + base_p1 + " + " + j + " * " +
		   to_string(m) + " + i] * " + par2 + "[" + base_p2 + " + i * " +
		   to_string(n) + " + " + k + "];\n}\n");
	code.prepend(type + " " + name + " = 0;\n" );
	return OCL_LAZY_DONT_PUSH_PREDS;
}
std::string MatMulImpl::generate_ocl_eager(FType res_type,
							  std::vector<FType> parameter_types) {
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
void MatMulImpl::execute_cpu(const FGraphNode *node,
							 std::vector<CPUResultData> predecessor_data,
							 void *__restrict__ result, size_t from,
							 size_t size) {
	BINARY_EXECUTE_IMPL
}
