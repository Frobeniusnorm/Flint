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
#include "../../flint.h"
#include "../operations/implementation.hpp"
#include "../utils.hpp"
#include "codegen.hpp"
#include <list>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

std::string generateEagerCode(FOperationType operation, FType res_type,
							  std::vector<FType> parameter_types,
							  std::string &kernel_name) {
	using namespace std;
	std::string type_info = to_string(res_type);
	for (FType t : parameter_types)
		type_info += to_string(t);
	kernel_name = string(fop_to_string[operation]) + type_info;
	string code =
		"#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n__kernel void " +
		kernel_name + "(__global " + typeString(res_type) +
		"* R, long num_entriesR";
	// generate parameters
	switch (operation) {
	case FMATMUL:
		for (int i = 0; i < 2; i++) {
			code += ", const __global " + typeString(parameter_types[i]) +
					"* P" + to_string(i) + ", long num_entries" + to_string(i) +
					", int dimensions" + to_string(i);
		}
		code += ", long l, long m, long n";
		break;
	case FREDUCE_MIN:
	case FREDUCE_MAX:
	case FREDUCE_SUM:
	case FREDUCE_MUL:
		code +=
			", const __global " + typeString(parameter_types[0]) +
			"* P0, const long num_entries0, const int dimensions0, const long "
			"it_dim0, const long shape_dim0";
		code += ", int reduce_dim";
		break;
	case FSLICE: {
		code += ", const __global " + typeString(parameter_types[0]) + "* P0";
		code += ", const long num_entries0, const int dimensions0";
		code += ", __constant long* acc_sizes, __constant long* acc_sizes_pred";
		code += ", __constant long* steps, const long start";
	} break;
	case FREPEAT: {
		code += ", const __global " + typeString(parameter_types[0]) + "* P0";
		code += ", const long num_entries0, const int dimensions0";
		code += ", __constant long* acc_sizes_d, __constant long* acc_sizes_s";
		code += ", __constant long* pred_shape";
	} break;
	case FTRANSPOSE: {
		code +=
			", const __global " + typeString(parameter_types[0]) +
			"* P0, const long num_entries0, const int dimensions0, __constant "
			"long* acc_sizes_d, __constant long* acc_sizes_s";
	} break;
	case FSET_INDEX: {
		code += ", const __global " + typeString(parameter_types[0]) +
				"* P0"
				", const long num_entries0, const int dimensions0"
				", const __global " +
				typeString(parameter_types[1]) +
				"* P1"
				", const long num_entries1, const int dimensions1 "
				", const __global " +
				typeString(parameter_types[2]) +
				"* P2"
				", const long num_entries2, const int dimensions2 "
				", const long acc_sizes_ax, const long op_shape_ax, const long "
				"c_shape_ax";
	} break;
	case FINDEX: {
		code += ", const __global " + typeString(parameter_types[0]) +
				"* P0"
				", const long num_entries0, const int dimensions0"
				", const __global " +
				typeString(parameter_types[1]) +
				"* P1"
				", const long num_entries1, const int dimensions1 "
				", const long acc_sizes_ax, const long op_shape_ax, const long "
				"a_shape_ax";
	} break;
	case FEXTEND: {
		code += ", const __global " + typeString(parameter_types[0]) + "* P0";
		code += ", const long num_entries0, const int dimensions0";
		code += ", __constant long* acc_sizes, __constant long* acc_sizes_pred";
		code += ", __constant long* steps, __constant long* start, __constant "
				"long* pred_shape";
	} break;
	case FCONVOLVE: {
		// acc_sizes, acc_sizes_pred, acc_sizes_kernel, steps
		code += ", const __global " + typeString(parameter_types[0]) + "* P0";
		code += ", const long num_entries0, const int dimensions0";
		code += ", const __global " + typeString(parameter_types[1]) + "* P1";
		code += ", const long num_entries1, const int dimensions1";
		code +=
			", __constant long* acc_sizes, __constant long* acc_sizes_pred, "
			"__constant long* acc_sizes_kernel";
		code += ", __constant int* steps";
	} break;
	case FGRADIENT_POOLING_MAX: {
		code +=
			", const __global " + typeString(parameter_types[0]) +
			"* P0"
			", const long num_entries0, const int dimensions0, const "
			"__global " +
			typeString(parameter_types[1]) +
			"* P1, const long num_entries1, const int dimensions1, const "
			"__global " +
			typeString(parameter_types[2]) +
			"* P2, const long num_entries2, const int dimensions2"
			", __constant long* acc_sizes_pred, "
			"__constant long* acc_sizes_kernel"
			", __constant long* acc_sizes, __constant long* acc_overlapping"
			", __constant int* steps, __constant long* op_shape, __constant "
			"long* kernel_shape";
	} break;
	case FGRADIENT_CONVOLVE1: {
		code +=
			", const __global " + typeString(parameter_types[0]) +
			"* P0"
			", const long num_entries0, const int dimensions0, const "
			"__global " +
			typeString(parameter_types[1]) +
			"* P1, const long num_entries1, const int dimensions1"
			", __constant long* acc_sizes_pred, "
			"__constant long* acc_sizes_kernel"
			", __constant long* acc_sizes, __constant long* acc_overlapping"
			", __constant int* steps, __constant long* op_shape, __constant "
			"long* kernel_shape";
	} break;
	case FGRADIENT_CONVOLVE2: {
		code +=
			", const __global " + typeString(parameter_types[0]) +
			"* P1"
			", const long num_entries1, const int dimensions1, const __global "
			"double* P2, const long num_entries2, const int dimensions2, "
			"const int dimensions0, "
			"__constant long* acc_sizes_pred, __constant long* "
			"acc_sizes_kernel, "
			"__constant long* acc_sizes_windows, __constant int* steps, "
			"__constant long* op_shape, __constant long* prev_adj_shape";
	} break;
	case FGEN_RANDOM: {
		code += ", const double time";
	} break;
	case FGEN_CONSTANT: {
		code += ", const " + typeString(res_type) + " constant_val";
	} break;
	case FGEN_ARANGE: {
		code += ", const long acc_sizes_ax, const long shape_ax";
	} break;
	case FCONCAT: {
		code += ", const __global " + typeString(parameter_types[0]) +
				"* P0, const long num_entries0, const __global " +
				typeString(parameter_types[1]) +
				"* P1, const long num_entries1, const long acc_size_last,"
				"const long shape_ax, const long a_shape_ax, const long "
				"b_shape_ax, const int ax";
	} break;
	case FPOOLING_MAX:
	case FPOOLING_SUM: {
		code +=
			", const __global " + typeString(parameter_types[0]) +
			"* P0"
			", const long num_entries0, const int dimensions0"
			", __constant long* acc_sizes_pred, __constant long* "
			"acc_sizes_kernel, __constant long* acc_sizes, __constant int* "
			"steps, const long pred_last_shape, const long kernel_num_elems";
	} break;
	case FSLIDING_WINDOW: {
		code += ", const __global " + typeString(parameter_types[0]) +
				"* P0"
				", const long num_entries0, const int dimensions0"
				", __constant long* acc_sizes_pred, __constant long* "
				"acc_sizes_win, __constant long* acc_sizes_rest, const long "
				"acc_sizes, __constant int* steps";
	} break;
	case FUNSLIDE_WINDOW: {
		code += ", const __global " + typeString(parameter_types[0]) +
				"* P0"
				", const long num_entries0, const int dimensions0"
				", __constant long* shapeR, __constant "
				"long* acc_sizes"
				", __constant long* shape0,  __constant long* acc_sizes_pred"
				", __constant long* acc_no_windows, __constant long* no_windows"
				", __constant int* steps";
	} break;
	default:
		for (int i = 0; i < parameter_types.size(); i++)
			code += ", const __global " + typeString(parameter_types[i]) +
					"* P" + to_string(i) + ", long num_entries" + to_string(i);
		break;
	}
	if (parameter_types.size() == 2)
		for (int i = 0; i < parameter_types.size(); i++)
			code += ", long inv_broad" + to_string(i);
	code += "){\nconst long index = get_global_id(0);\n";
	// generate code
	code +=
		OperationImplementation::implementations[operation]->generate_ocl_eager(
			res_type, parameter_types);
	code += "\n}\n";
	return code;
}
