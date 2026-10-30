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
#include "../utils.hpp"
#include <random>
#include <climits>
#include <cstdint>
using namespace std;
void GenRandomImpl::execute_cpu(const FGraphNode *node,
								std::vector<CPUResultData> predecessor_data,
								void *__restrict__ result, size_t from,
								size_t size) {
	double seed = ((double *)node->operation.additional_data)[0];
	std::minstd_rand0 g1(seed * 1000 + from);
	const bool single = node->operation.data_type == F_FLOAT32;
	for (size_t i = from; i < from + size; i++) {
		const double val = (g1() % 100000000) / 100000000.0;
		if (single)
			((float *)result)[i] = (float)val;
		else
			((double *)result)[i] = val;
	}
}
// integer hash with good avalanche (lowbias32 by Chris Wellons)
static uint32_t hash32(uint32_t x) {
	x ^= x >> 16;
	x *= 0x7feb352dU;
	x ^= x >> 15;
	x *= 0x846ca68bU;
	x ^= x >> 16;
	return x;
}
int GenRandomImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
									 OCLLazyCodegenState &compiler_state) {
	const FType dt = node->operation.data_type;
	const string type = type_string(dt);
	double seed = ((double *)node->operation.additional_data)[0];
	// the seed differs per node, as an argument it keeps the code cacheable.
	// It is hashed here once so that the kernel only has to hash the offset
	const uint32_t hs = hash32((uint32_t)(seed * 100));
	const string sn = compiler_state.addScalar(F_INT32, (int32_t)hs);
	// hashing the integer offset keeps neighbouring entries independent, a
	// float offset can't distinguish them anymore beyond 2^24
	const IndexExpr flat = compiler_state.flatten(
		compiler_state.index_map, node->operation.shape,
		node->operation.dimensions);
	const string offset = compiler_state.to_code(flat);
	const string h = name + "_h";
	string code = type + " " + name + " = 0;\n{\n";
	if (flat.bound.first < 0 || flat.bound.second > UINT_MAX)
		code += " const ulong " + h + "_l = " + offset + ";\n uint " + h +
				" = (uint)" + h + "_l ^ (uint)(" + h + "_l >> 32);\n";
	else
		code += " uint " + h + " = " + offset + ";\n";
	code += " " + h + " ^= (uint)" + sn + ";\n";
	for (const string &step : std::initializer_list<string>
		 {"^= " + h + " >> 16", "*= 0x7feb352dU", "^= " + h + " >> 15",
		  "*= 0x846ca68bU", "^= " + h + " >> 16"})
		code += " " + h + " " + step + ";\n";
	// the upper 24 bits fit exactly into a float's mantissa, so the result
	// stays below 1
	code += " " + name + " = (" + h + " >> 8) * " +
			(dt == F_FLOAT32 ? "0x1p-24f" : "0x1p-24") + ";\n}\n";
	compiler_state.code.prepend(code);
	return 0;
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
	const string type = type_string(node->operation.data_type);
	const unsigned int ax = ((unsigned int *)node->operation.additional_data)[0];
	// the value is the coordinate of its axis, which the index map holds
	compiler_state.code.prepend(
		"const " + type + " " + name + " = " +
		compiler_state.to_code(
			compiler_state.index_map.expr_per_dim[ax]) +
		";\n");
	return 0;
}
