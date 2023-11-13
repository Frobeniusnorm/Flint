#include "../../flint.h"
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
	case FGRADIENT_CONVOLVE1: {
		code +=
			", const __global " + typeString(parameter_types[0]) +
			"* P1"
			", const long num_entries1, const int dimensions1, const __global "
			"double* P2, const long num_entries2, const int dimensions2"
			", const int dimensions0"
			", __constant long* acc_sizes_pred, "
			"__constant long* acc_sizes_kernel"
			", __constant long* acc_sizes"
			", __constant int* steps, __constant long* shape1";
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
	case FSLIDE: {
		// acc_sizes, acc_sizes_pred, acc_sizes_kernel, steps
		code += ", const __global " + typeString(parameter_types[0]) + "* P0";
		code += ", const long num_entries0, const int dimensions0";
		code += ", const __global " + typeString(parameter_types[1]) + "* P1";
		code += ", const long num_entries1, const int dimensions1";
		code += ", __constant long* acc_sizes_pred, "
				"__constant long* acc_sizes_kernel";
		code += ", __constant int* steps, __constant long* shape0, __constant "
				"long* shape1";
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
	switch (operation) {
	case FADD:
		code += "if(index >= num_entries0 && index >= num_entries1) "
				" return;\n"
				"R[index] = P0[(index/inv_broad0)%num_entries0] + "
				"P1[(index/inv_broad1)%num_entries1];";
		break;
	case FSUB:
		code += "if(index >= num_entries0 && index >= num_entries1) "
				"return;\nR[index] = "
				"P0[(index/inv_broad0)%num_entries0] - "
				"P1[(index/inv_broad1)%num_entries1];";
		break;
	case FMUL:
		code += "if(index >= num_entries0 && index >= num_entries1) "
				"return;\nR[index] = "
				"P0[(index/inv_broad0)%num_entries0] * "
				"P1[(index/inv_broad1)%num_entries1];";
		break;
	case FDIV:
		code += "if(index >= num_entries0 && index >= num_entries1) "
				"return;\nR[index] = "
				"P0[(index/inv_broad0)%num_entries0] / "
				"P1[(index/inv_broad1)%num_entries1];";
		break;
	case FPOW: {
		code += "if(index >= num_entries0 && index >= num_entries1) return;\n";
		string type = typeString(res_type);
		if ((parameter_types[0] == F_FLOAT32 ||
			 parameter_types[0] == F_FLOAT64) &&
			(parameter_types[1] == F_FLOAT32 ||
			 parameter_types[1] == F_FLOAT64))
			code += "R[index] = pow((" + type +
					")P0[(index/inv_broad0)%num_entries0], (" + type +
					")P1[(index/inv_broad1)%num_entries1]);";
		else if (parameter_types[0] == F_INT64 &&
				 (parameter_types[1] == F_INT32 ||
				  parameter_types[1] == F_INT64))
			code += "R[index] "
					"= (long)pown((double)P0[(index/inv_broad0)%num_entries0], "
					"(int)P1[(index/inv_broad1)%num_entries1]);";
		else if (parameter_types[0] == F_INT32 &&
				 (parameter_types[1] == F_INT32 ||
				  parameter_types[1] == F_INT64))
			code += "R[index] = "
					"(int)pown((float)P0[(index/inv_broad0)%num_entries0], "
					"(int)P1[(index/inv_broad1)%num_entries1]);";
		else
			code += "R[index] = "
					"pow((double)P0[(index/inv_broad0)%num_entries0], "
					"(double)P1[(index/inv_broad1)%num_entries1]);";
	} break;
	case FNEG:
		code += "if(index >= num_entries0) return;\n"
				"R[index] = "
				"-P0[index];";
		break;
	case FLOG: {
		std::string conv = parameter_types[0] == F_INT32   ? "(float)"
						   : parameter_types[0] == F_INT64 ? "(double)"
														   : "";
		code += "if(index >= num_entries0) return;\n"
				"R[index] = "
				"log(" +
				conv + "P0[index]);";
	} break;
	case FSIGN:
		code += "if(index >= num_entries0) return;\n"
				"R[index] = "
				"P0[index] >= 0 ? 1 : -1;";
		break;
	case FEVEN:
		code += "if(index >= num_entries0) return;\n"
				"R[index] = "
				"P0[index] % 2 == 0 ? 1 : 0;";
		break;
	case FMATMUL: {
		code +=
			"if(index >= num_entriesR) return;\n" + typeString(res_type) +
			" res = 0;\n"
			"long j = (index % (l * n)) / n;\n"
			"long k = (index % (l * n)) % n;\n"
			"long base_p0 = dimensions0 > 2 ? (index / (l * n)) * (l * m) : "
			"0;\n"
			"long base_p1 = dimensions1 > 2 ? (index / (l * n)) * (m * n) : "
			"0;\n"
			"for(int i = 0; i < m; i++){\n res += P0[base_p0 + j * m + i] * "
			"P1[base_p1 + i * n + k];\n}"
			"R[index] = res;\n";
		break;
	}
	case FABS:
		code += "if(index >= num_entries0) return;\n"
				"R[index] = "
				"P0[index] < 0 ? -P0[index] : P0[index];";
		break;
	case FSQRT:
		code += "if(index >= num_entries0) return;\n"
				"R[index] = "
				"sqrt(P0[index]);";
		break;
	case FEXP:
		code += "if(index >= num_entries0) return;\n"
				"R[index] = "
				"exp(P0[index]);";
		break;
	case FSIN:
		code += "if(index >= num_entries0) return;\n"
				"R[index] = "
				"sin(P0[index]);";
		break;
	case FCOS:
		code += "if(index >= num_entries0) return;\n"
				"R[index] = "
				"cos(P0[index]);";
		break;
	case FTAN:
		code += "if(index >= num_entries0) return;\n"
				"R[index] = "
				"tan(P0[index]);";
		break;
	case FASIN:
		code += "if(index >= num_entries0) return;\n"
				"R[index] = "
				"asin(P0[index]);";
		break;
	case FACOS:
		code += "if(index >= num_entries0) return;\n"
				"R[index] = "
				"acos(P0[index]);";
		break;
	case FATAN:
		code += "if(index >= num_entries0) return;\n"
				"R[index] = "
				"atan(P0[index]);";
		break;
	case FLOG2: {
		std::string conv = parameter_types[0] == F_INT32   ? "(float)"
						   : parameter_types[0] == F_INT64 ? "(double)"
														   : "";
		code += "if(index >= num_entries0) return;\n"
				"R[index] = "
				"log2(" +
				conv + "P0[index]);";
	} break;
	case FLOG10: {
		std::string conv = parameter_types[0] == F_INT32   ? "(float)"
						   : parameter_types[0] == F_INT64 ? "(double)"
														   : "";
		code += "if(index >= num_entries0) return;\n"
				"R[index] = "
				"log10(" +
				conv + "P0[index]);";
	} break;
		// case FLATTEN:
		// case FRESHAPE:
	case FCONVERSION:
		code += "if(index >= num_entries0) return;\n";
		code += "R[index] = (" + typeString(res_type) + ")P0[index];";
		break;
	case FMIN:
		code += "if(index >= num_entries0 && index >= num_entries1) return;\n";
		code += typeString(parameter_types[0]) +
				" a = P0[(index/inv_broad0)%num_entries0];\n";
		code += typeString(parameter_types[1]) +
				" b = P1[(index/inv_broad1)%num_entries1];\n";
		code += "R[index] = a < b ? a : b;";
		break;
	case FMAX:
		code += "if(index >= num_entries0 && index >= num_entries1) return;\n";
		code += typeString(parameter_types[0]) +
				" a = P0[(index/inv_broad0)%num_entries0];\n";
		code += typeString(parameter_types[1]) +
				" b = P1[(index/inv_broad1)%num_entries1];\n";
		code += "R[index] = a > b ? a : b;";
		break;
	case FLESS:
		code += "if(index >= num_entries0 && index >= num_entries1) return;\n";
		code += typeString(parameter_types[0]) +
				" a = P0[(index/inv_broad0)%num_entries0];\n";
		code += typeString(parameter_types[1]) +
				" b = P1[(index/inv_broad1)%num_entries1];\n";
		code += "R[index] = a < b ? 1 : 0;";
		break;
	case FEQUAL:
		code += "if(index >= num_entries0 && index >= num_entries1) return;\n";
		code += typeString(parameter_types[0]) +
				" a = P0[(index/inv_broad0)%num_entries0];\n";
		code += typeString(parameter_types[1]) +
				" b = P1[(index/inv_broad1)%num_entries1];\n";
		code += "R[index] = a + " + epsilonForType(parameter_types[0]) +
				" >= b && a <= b + " + epsilonForType(parameter_types[1]) +
				" ? 1 : 0;";
		break;
	case FGREATER:
		code += "if(index >= num_entries0 && index >= num_entries1) return;\n";
		code += typeString(parameter_types[0]) +
				" a = P0[(index/inv_broad0)%num_entries0];\n";
		code += typeString(parameter_types[1]) +
				" b = P1[(index/inv_broad1)%num_entries1];\n";
		code += "R[index] = a > b ? 1 : 0;";
		break;
	case FREDUCE_MIN:
	case FREDUCE_MAX:
	case FREDUCE_SUM:
	case FREDUCE_MUL:
		// it_dim, shape_dim
		code += "if(index >= num_entries0) return;\n";
		code += typeString(res_type) + " res = ";
		switch (operation) {
		case FREDUCE_SUM:
			code += "0";
			break;
		case FREDUCE_MUL:
			code += "1";
			break;
		case FREDUCE_MIN:
			code += "P0[(index / it_dim0) * it_dim0 * shape_dim0 + index % "
					"it_dim0]";
			break;
		case FREDUCE_MAX:
			code += "P0[(index / it_dim0) * it_dim0 * shape_dim0 + index % "
					"it_dim0]";
			break;
		default:
			break;
		}
		code += ";\n";
		code += "for(long i = 0; i < shape_dim0; i++){\n"
				" const " +
				typeString(res_type) +
				" curr = P0[(index / it_dim0) * it_dim0 * shape_dim0 + index % "
				"it_dim0 "
				"+ i * it_dim0];\n";
		switch (operation) {
		case FREDUCE_SUM:
			code += " res += curr;";
			break;
		case FREDUCE_MUL:
			code += " res *= curr;";
			break;
		case FREDUCE_MIN:
			code += " res = res < curr ? res : curr;";
			break;
		case FREDUCE_MAX:
			code += " res = res >= curr ? res : curr;";
			break;
		default:
			break;
		}
		code += "\n}R[index] = res;\n";
		break;
	case FTRANSPOSE:
		code += "if(index >= num_entries0) return;\n"
				"long src_index = 0;\n"
				"int i = index;\n"
				"for(int dim = 0; dim < dimensions0; dim++){\n"
				" int curr_idx = i / acc_sizes_d[dim];\n"
				" i %= acc_sizes_d[dim];\n"
				" src_index += curr_idx * acc_sizes_s[dim];\n}\n"
				"R[index] = P0[src_index];\n";
		break;
	case FSET_INDEX:
		code += "if(index >= num_entriesR) return;\n"
				"const int axis = dimensions2 - 1;\n"
				"const long base = index / (acc_sizes_ax * op_shape_ax);\n"
				"const long rest = index % acc_sizes_ax;\n"
				"const long axi = (index / acc_sizes_ax) % op_shape_ax;\n"
				"const long base_ind = base * c_shape_ax;\n"
				"R[index] = 0;\n"
				"int found_something = false;\n"
				"for (long j = base_ind; j < base_ind + c_shape_ax; j++) {\n"
				" const long ind = (long) P2[j];\n"
				" if(ind == axi){"
				"   R[index] += P1[j * acc_sizes_ax + rest];\n"
				"   found_something = true;\n"
				" }\n"
				"}\n"
				"if(!found_something) R[index] = P0[index];\n";
		break;
	case FINDEX:
		code += "if(index >= num_entriesR) return;\n"
				"const int axis = dimensions1 - 1;\n"
				"const long base = index / (acc_sizes_ax * op_shape_ax);\n"
				"const long rest = index % acc_sizes_ax;\n"
				"const long ind = (long) P1[index / acc_sizes_ax];\n"
				"R[index] = P0[(base * acc_sizes_ax * a_shape_ax) + (ind * "
				"acc_sizes_ax) + rest];\n";
		break;
	case FSLICE:
		code += "if(index >= num_entriesR) return;\n"
				"long j = start;\n"
				"for (int d = 0; d < dimensions0; d++){\n"
				" long di = (d == 0 ? index : index % acc_sizes[d - 1]) /"
				"acc_sizes[d];\n"
				" j += di * steps[d] * acc_sizes_pred[d];\n}\n"
				"R[index] = P0[j];\n";
		break;
	case FREPEAT:
		code +=
			"if(index >= num_entriesR) return;\n"
			"long src_index = 0;\n"
			"int i = index;\n"
			"for (int dim = 0; dim < dimensions0; dim++){\n"
			" int curr = i / acc_sizes_d[dim];\n"
			" i %= acc_sizes_d[dim];\n"
			" src_index += (curr % pred_shape[dim]) * acc_sizes_s[dim];\n}\n"
			"R[index] = P0[src_index];\n";
		break;
	case FEXTEND:
		code += "if(index >= num_entriesR) return;\n"
				"long j = 0;\n"
				"int set_zero = 0;\n"
				"for(int d = 0; d < dimensions0; d++){\n"
				" long step = steps[d];\n"
				" int inv = step < 0;\n"
				" if(inv) step = -step;\n"
				" long di = (d == 0 ? index : index % acc_sizes[d - 1]) / "
				"acc_sizes[d];\n"
				" if(di < start[d]){\n"
				"  set_zero = 1;\n  break;\n }\n"
				" di -= start[d];\n"
				" if(di % step != 0){\n  set_zero = 1;\n  break;\n }\n"
				" di /= step;\n"
				" if(di >= pred_shape[d]){\n"
				"  set_zero = 1;\n  break;\n }\n"
				" if(inv) di = pred_shape[d] - di - 1;\n"
				" j += di * acc_sizes_pred[d];\n}\n"
				"R[index] = set_zero ? 0 : P0[j];";
		break;
	case FCONCAT:
		code +=
			"if(index >= num_entriesR) return;\n"
			"long sx = index / acc_size_last;\n"
			"long sc = ax > 0 ? sx % shape_ax : sx;\n"
			"if(sc < a_shape_ax){\n"
			" long ai = (sx / shape_ax) * acc_size_last * a_shape_ax + sc * "
			"acc_size_last + index % acc_size_last;\n"
			" R[index] = P0[ai];\n"
			"}else{\n"
			" long bi = (sx / shape_ax) * acc_size_last * b_shape_ax + (sc - "
			"a_shape_ax) * "
			"acc_size_last + index % acc_size_last;\n"
			" R[index] = P1[bi];\n"
			"}";
		break;
	case FCONVOLVE:
		code +=
			"if(index >= num_entriesR) return;\n"
			"int multi_filter = dimensions0 != dimensions1;\n"
			"long j = 0;\n"
			"for(int d = 0; d < dimensions0 - 1; d++){\n"
			" long di = (d == 0 ? index : index % acc_sizes[d - 1]) / "
			"acc_sizes[d];\n"
			" j += di * steps[d] * acc_sizes_pred[d];\n"
			"}\n"
			"long kernel_offset = 0;\n"
			"if(multi_filter){\n"
			" long fi = (index % acc_sizes[dimensions0 - 2]) / "
			"acc_sizes[dimensions0 - 1];\n"
			" kernel_offset = fi * acc_sizes_kernel[0];\n"
			"}\n" +
			typeString(res_type) +
			" res = 0;\n"
			"const long kernel_num_elems = multi_filter ? acc_sizes_kernel[0] "
			": "
			"num_entries1;\n"
			"for(long k = 0; k < kernel_num_elems; k++){\n"
			" bool set_zero = false;\n"
			" long o = 0;\n"
			" const int last_dim = multi_filter ? dimensions1 - 1 : "
			"dimensions1;\n"
			" for(int d = 0; d < last_dim; d++){\n"
			"  const int kn_d = multi_filter ? d + 1 : d;\n"
			"  long di = d == last_dim ? 0 : (d == 0 ? index : index % "
			"acc_sizes[d - 1]) / "
			"acc_sizes[d];\n"
			"  long dk = (kn_d == 0 ? k : k % acc_sizes_kernel[kn_d - 1]) / "
			"acc_sizes_kernel[kn_d];\n"
			"  if(d < dimensions0 - 1)\n"
			"   if(((di * steps[d]) + dk) * acc_sizes_pred[d] >= num_entries0 "
			"||\n"
			"        (d > 0 && ((di * steps[d]) + dk) * acc_sizes_pred[d] >= \n"
			"acc_sizes_pred[d - 1])) {\n"
			"    set_zero = true; break;\n}\n"
			"  o += dk * acc_sizes_pred[d];\n"
			" }\n"
			" if (set_zero) continue;\n"
			" res += P1[k + kernel_offset] * P0[j + o];\n"
			"}\n"
			"R[index] = res;";
		break;
	case FGEN_CONSTANT: {
		code += "if(index >= num_entriesR) return;\n"
				"R[index] = constant_val;\n";
	} break;
	case FGEN_RANDOM: {
		code += "if(index >= num_entriesR) return;\n"
				"const double v = sin(index + time) * 43758.5453123;\n"
				"R[index] = min(v - floor(v), 0.99999);\n";
	} break;
	case FGEN_ARANGE: {
		code += "if(index >= num_entriesR) return;\n"
				"const long i = (index / acc_sizes_ax) % shape_ax\n;"
				"R[index] = i;\n";
	} break;
	case FGRADIENT_CONVOLVE2:
		code +=
			"if(index >= num_entriesR) return;\n"
			"const bool multifilter = dimensions0 > dimensions1;\n"
			"const long windows = acc_sizes_windows[0] * prev_adj_shape[0];\n"
			"const long num_elems_kernel = multifilter ? acc_sizes_kernel[0] : "
			"acc_sizes_kernel[0] * op_shape[0];\n"
			"const int num_filter = multifilter ? op_shape[0] : 1;\n"
			"const long f = multifilter ? index / num_elems_kernel : 0;\n"
			"long a_offset = 0;\n"
			"for(int j = multifilter ? 1 : 0; j < dimensions0; j++){\n"
			" const long ki = (index / acc_sizes_kernel[j]) % op_shape[j];\n"
			" a_offset += ki * acc_sizes_pred[multifilter ? j - 1 : j];\n"
			"}\n"
			"R[index] = 0;\n"
			"for(long w = 0; w < windows; w++){\n"
			" long a = 0;"
			" for(int j = 0; j < (multifilter ? dimensions2 - 1 : "
			"dimensions2); "
			"j++){\n"
			"  const long wj = (w / acc_sizes_windows[j]) % "
			"prev_adj_shape[j];\n"
			"  a += wj * acc_sizes_pred[j] * steps[j];\n"
			" }\n"
			" R[index] += P1[a + a_offset] * P2[w * num_filter + f];\n"
			"}\n";
		break;
	case FGRADIENT_CONVOLVE1:
		code +=
			"if(index >= num_entriesR) return;\n"
			"long k = 0;\n"
			"int in_steps = 1;\n"
			"for(int d = dimensions0 - 1; d >= 0; d--){\n"
			" long di = (d == 0 ? index : index % acc_sizes_pred[d - 1]) / "
			"acc_sizes_pred[d];\n"
			" long dk = d == dimensions0 - 1 ? di : di % steps[d];\n"
			" if(dk >= shape1[d]){\n"
			"  in_steps = 0;\n"
			"  break;\n"
			" }\n"
			" k += dk * acc_sizes_kernel[d];\n"
			"}\n" +
			typeString(res_type) +
			" res = 0;\n"
			"if(in_steps)\n"
			" while(k < num_entries1){\n"
			"  long i_conv = 0;\n"
			"  for(int d = 0; d < dimensions0 - 2; d++){\n"
			"   long dk = (d == 0 ? k : k % acc_sizes_kernel[d - 1]) / "
			"acc_sizes_kernel[d];\n"
			"   long di = (d == 0 ? index : index % acc_sizes_pred[d - 1]) / "
			"acc_sizes_pred[d];\n"
			"   i_conv += ((di - dk) / steps[d]) * acc_sizes[d];\n"
			"  }\n"
			"  if (i_conv < num_entries2)\n"
			"   res += P1[k] * P2[i_conv];\n"
			"  long step = 0;\n"
			"  for(int d = dimensions0 - 2; d >= 0; d--) {\n"
			"   long dk = (d == 0 ? k : k % acc_sizes_kernel[d - 1]) / "
			"acc_sizes_kernel[d];\n"
			"   long di = (d == 0 ? index : index % acc_sizes_pred[d - 1]) / "
			"acc_sizes_pred[d];\n"
			"   if(dk + steps[d] < shape1[d] && di >= dk + steps[d]){\n"
			"    step += steps[d] * acc_sizes_kernel[d];\n"
			"    break;\n"
			"   }else{\n"
			"    step -= (dk - (di % steps[d])) * acc_sizes_kernel[d];\n"
			"   }\n"
			"  }\n"
			"  if(step <= 0) break;\n"
			"  k += step;\n"
			" }\n"
			"R[index] = res;";
		break;
	case FSLIDE:
		code +=
			"if(index >= num_entriesR) return;\n"
			"long a = 0;\n"
			"for(int d = dimensions1 - 1; d >= 0; d--){\n"
			" long di = (d == 0 ? index : index % acc_sizes_kernel[d - 1]) / "
			"acc_sizes_kernel[d];\n"
			" a += di * acc_sizes_pred[d];\n}\n" +
			typeString(res_type) +
			" res = 0;\n"
			"while(a < num_entries0){\n"
			" long step = 0;\n"
			" res += P0[a] * P1[index];\n"
			" for(int d = dimensions0 - 2; d >= 0; d--){\n"
			"  long da = (d == 0 ? a : a % acc_sizes_pred[d-1]) / "
			"acc_sizes_pred[d];\n"
			"  long di = (d == 0 ? index : index % acc_sizes_kernel[d - 1]) / "
			"acc_sizes_kernel[d];\n"
			"  if(da + (shape1[d] - di - 1) + steps[d] < shape0[d]){\n"
			"   step += steps[d] * acc_sizes_pred[d];\n"
			"   break;\n  }else{\n"
			"   long di = (d == 0 ? index : index % acc_sizes_kernel[d - 1]) / "
			"acc_sizes_kernel[d];\n"
			"   step -= (da - di) * acc_sizes_pred[d];\n  }\n }\n"
			" if (step <= 0) break;\n"
			" a += step;\n"
			"}\nR[index] = res;";
		break;
	case FPOOLING_MAX:
	case FPOOLING_SUM:
		code += "if(index >= num_entriesR) return;\n"
				"long j = 0;\n"
				"for(int d = 0; d < dimensions0 - 1; d++){\n"
				" const long di = (d == 0 ? index : index%acc_sizes[d - 1]) / "
				"acc_sizes[d];\n"
				" j += di * steps[d] * acc_sizes_pred[d];\n"
				"}\n" +
				typeString(res_type) + " res = ";
		if (operation == FPOOLING_SUM)
			code += "0";
		else
			code += minForType(res_type);
		code += ";\n"
				"for(long k = 0; k < kernel_num_elems; k++){\n"
				" int set_zero = false;\n"
				" long o = 0;\n"
				" for(int d = 0; d < dimensions0 - 1; d++){"
				"  const long dk = (d == 0 ? k : k%acc_sizes_kernel[d - 1]) / "
				"acc_sizes_kernel[d];\n"
				"  o += dk * acc_sizes_pred[d];\n"
				" }"
				" for(long ld = 0; ld < pred_last_shape; ld++){";
		if (operation == FPOOLING_SUM) {
			code += "  res += P0[j + o + ld];\n";
		} else {
			code += "  res = max(res, P0[j + o + ld]);\n";
		}
		code += " }\n"
				"}\n"
				"R[index] = res;\n";
		break;
	case FSLIDING_WINDOW:
		code += "if(index >= num_entriesR) return;\n"
				"long wi = index / acc_sizes;\n"
				"long rest = index % acc_sizes;\n"
				"long offset = 0, base = 0;\n"
				"for(int d = 0; d < dimensions0; d++){\n"
				" long local_wi = wi / acc_sizes_win[d];\n"
				" long local_base = local_wi * steps[d];\n"
				" base += local_base * acc_sizes_pred[d];\n"
				" wi %= acc_sizes_win[d];\n"
				" long local_ri = rest / acc_sizes_rest[d];\n"
				" offset += local_ri * acc_sizes_pred[d];\n"
				" rest %= acc_sizes_rest[d];\n"
				"}\n"
				"R[index] = P0[base + offset];\n";
		break;
	case FUNSLIDE_WINDOW:
		code +=
			"if(index >= num_entriesR) return;\n"
			"R[index] = 0;\n"
			"long first_w = 0;\n"
			"long last_w = 0;\n"
			"for (int d = 0; d < dimensions0 - 1; d++) {\n"
			" const long id = (index / acc_sizes[d]) % shapeR[d];\n"
			" const long wdf = max(0l, (id - shape0[d + 1] + 1)) / steps[d];\n"
			" const long wfl = id / steps[d];\n"
			" first_w += wdf * acc_no_windows[d];\n"
			" last_w += wfl * acc_no_windows[d];\n"
			"}\n"
			"for (long w = first_w; w <= last_w;) {\n"
			" int contained = true;\n"
			" long wi = 0;\n"
			" long wpp = 0;\n"
			" for (int d = dimensions0 - 2; d >= 0; d--) {\n"
			"  const long wd = (w/acc_no_windows[d]) % no_windows[d];\n"
			"  const long w_start = wd * steps[d]\n;"
			"  const long id = (index / acc_sizes[d]) % shapeR[d];\n"
			"  if (id >= w_start && id < w_start + shape0[d + 1])\n"
			"   wi += (id - w_start) * acc_sizes_pred[d + 1];\n"
			"  else {\n"
			"   contained = false;\n"
			"   wpp += acc_no_windows[d];\n"
			"  }\n"
			" }\n"
			" if (contained) {\n"
			"   R[index] += P0[wi + w * acc_sizes_pred[0]];\n"
			"   wpp = 1;\n"
			" }\n"
			" w += wpp;\n"
			"}\n";
		break;
	}
	code += "\n}\n";
	return code;
}
