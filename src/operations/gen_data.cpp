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
#include "gen_data.hpp"

using namespace std;

void GenRandomImpl::execute_cpu(const FGraphNode *node,
								std::vector<CPUResultData> predecessor_data,
								void *__restrict__ result, size_t from,
								size_t size) {
	double seed = ((double *)node->operation.additional_data)[0];
	for (size_t i = from; i < from + size; i++) {
		double v = sin(i + seed) * 43758.5453123;
		((double *)result)[i] = std::min(v - floor(v), 0.99999);
	}
}
int GenRandomImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
									 OCLLazyCodegenState &compiler_state) {
	const string type = typeString(node->operation.data_type);
	double seed = ((double *)node->operation.additional_data)[0];
	compiler_state.code.prepend(type + " " + name + " = 0;\n{\n " + name +
								" = sin(index + " + std::to_string(seed) +
								") * 43758.5453123;\n " + name + " = min(" +
								name + " - floor(" + name +
								"), 0.99999);\n"
								"}\n");
	return 0;
}
std::string
GenRandomImpl::generate_ocl_eager(FType res_type,
								  std::vector<FType> parameter_types) {
  return "if(index >= num_entriesR) return;\n"
				"const double v = sin(index + time) * 43758.5453123;\n"
				"R[index] = min(v - floor(v), 0.99999);\n";
}

template <typename T>
void GenConstantImpl::zeroary_expression(const FGraphNode *node,
										 T *__restrict__ result, size_t from,
										 size_t size) {
	T value = ((T *)node->operation.additional_data)[0];
	for (size_t i = from; i < from + size; i++)
		result[i] = value;
}
int GenConstantImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
									   OCLLazyCodegenState &compiler_state) {
	flogging(F_ERROR, "Constant Generation should not be implemented in OpenCL "
					  "code generation!");
	return 0;
}
std::string
GenConstantImpl::generate_ocl_eager(FType res_type,
									std::vector<FType> parameter_types) {
  return "if(index >= num_entriesR) return;\n"
				"R[index] = constant_val;\n";
}
void GenConstantImpl::execute_cpu(const FGraphNode *node,
								  std::vector<CPUResultData> predecessor_data,
								  void *__restrict__ result, size_t from,
								  size_t size) {
	ZEROARY_EXECUTE_IMPL
}
void GenArangeImpl::execute_cpu(const FGraphNode *node,
								std::vector<CPUResultData> predecessor_data,
								void *__restrict__ result, size_t from,
								size_t size) {
	unsigned int ax = ((unsigned int *)node->operation.additional_data)[0];
	size_t acc_sizes_ax = 1;
	for (unsigned int i = ax + 1; i < node->operation.dimensions; i++)
		acc_sizes_ax *= node->operation.shape[i];
	for (size_t i = from; i < from + size; i++)
		((long *)result)[i] = (i / acc_sizes_ax) % node->operation.shape[ax];
}
int GenArangeImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
									 OCLLazyCodegenState &compiler_state) {
	const string type = typeString(node->operation.data_type);
	unsigned int ax = ((unsigned int *)node->operation.additional_data)[0];
	size_t acc_sizes_ax = 1;
	for (unsigned int i = ax + 1; i < node->operation.dimensions; i++)
		acc_sizes_ax *= node->operation.shape[i];
	compiler_state.code.prepend("const " + type + " " + name + " = (index/" +
								to_string(acc_sizes_ax) + ")%" +
								to_string(node->operation.shape[ax]) + ";\n");
	return 0;
}
std::string
GenArangeImpl::generate_ocl_eager(FType res_type,
								  std::vector<FType> parameter_types) {
  return "if(index >= num_entriesR) return;\n"
				"const long i = (index / acc_sizes_ax) % shape_ax\n;"
				"R[index] = i;\n";
}
