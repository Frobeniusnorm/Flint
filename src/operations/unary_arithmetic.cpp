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
#include "unary_arithmetic.hpp"
#include "../utils.hpp"

using namespace std;

FGraphNode *NegImpl::local_gradient(FGraphNode *y, int dx_i,
									FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	if (0 == dx_i) {
		return fneg(prev_adj);
	} else
		return nullptr;
}
template <typename T>
void NegImpl::unary_expression(T *__restrict__ result,
							   const T *__restrict__ data, size_t from,
							   size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] =
			-data[curr->predecessors[0]->operation.op_type == FGEN_CONSTANT
					  ? 0
					  : i];
}
int NegImpl::generate_ocl_lazy(const FGraphNode *node, string name,
							   OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = -v" + to_string(compiler_state.variable_index + 1) + ";\n");
	return 0;
}
std::string NegImpl::generate_ocl_eager(FType res_type,
										std::vector<FType> parameter_types) {
	return "if(index >= num_entriesR) return;\n"
		   "R[index] = "
		   "-P0[p0_is_constant ? 0 : index];";
}
void NegImpl::execute_cpu(const FGraphNode *node,
						  vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from,
						  size_t size){UNARY_EXECUTE_MONOTON_IMPL}

FGraphNode *LogImpl::local_gradient(FGraphNode *y, int dx_i,
									FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	if (0 == dx_i)
		return fdiv(prev_adj, a);
	else
		return nullptr;
}
template <typename T, typename A>
void LogImpl::unary_expression(T *__restrict__ result,
							   const A *__restrict__ data, size_t from,
							   size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] =
			log(data[curr->predecessors[0]->operation.op_type == FGEN_CONSTANT
						 ? 0
						 : i]);
}
int LogImpl::generate_ocl_lazy(const FGraphNode *node, string name,
							   OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = log(v" + to_string(compiler_state.variable_index + 1) + ");\n");
	return 0;
}
std::string LogImpl::generate_ocl_eager(FType res_type,
										std::vector<FType> parameter_types) {
	const string conv = parameter_types[0] == F_INT32	? "(float)"
						: parameter_types[0] == F_INT64 ? "(double)"
														: "";
	return "if(index >= num_entriesR) return;\n"
		   "R[index] = "
		   "log(" +
		   conv + "P0[p0_is_constant ? 0 : index]);";
}
void LogImpl::execute_cpu(const FGraphNode *node,
						  vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from,
						  size_t size){UNARY_EXECUTE_IMPL}

FGraphNode *Log2Impl::local_gradient(FGraphNode *y, int dx_i,
									 FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	if (0 == dx_i)
		return fdiv(prev_adj, fmul(a, log(2.0)));
	else
		return nullptr;
}
template <typename T, typename A>
void Log2Impl::unary_expression(T *__restrict__ result,
								const A *__restrict__ data, size_t from,
								size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] =
			log2(data[curr->predecessors[0]->operation.op_type == FGEN_CONSTANT
						  ? 0
						  : i]);
}
int Log2Impl::generate_ocl_lazy(const FGraphNode *node, string name,
								OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = log2(v" + to_string(compiler_state.variable_index + 1) + ");\n");
	return 0;
}
std::string Log2Impl::generate_ocl_eager(FType res_type,
										 std::vector<FType> parameter_types) {
	const string conv = parameter_types[0] == F_INT32	? "(float)"
						: parameter_types[0] == F_INT64 ? "(double)"
														: "";
	return "if(index >= num_entriesR) return;\n"
		   "R[index] = "
		   "log2(" +
		   conv + "P0[p0_is_constant ? 0 : index]);";
}
void Log2Impl::execute_cpu(const FGraphNode *node,
						   vector<CPUResultData> predecessor_data,
						   void *__restrict__ result, size_t from,
						   size_t size){UNARY_EXECUTE_IMPL}

FGraphNode *Log10Impl::local_gradient(FGraphNode *y, int dx_i,
									  FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	if (0 == dx_i)
		return fdiv(prev_adj, fmul(a, log(10.0)));
	else
		return nullptr;
}
template <typename T, typename A>
void Log10Impl::unary_expression(T *__restrict__ result,
								 const A *__restrict__ data, size_t from,
								 size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] =
			log10(data[curr->predecessors[0]->operation.op_type == FGEN_CONSTANT
						   ? 0
						   : i]);
}
int Log10Impl::generate_ocl_lazy(const FGraphNode *node, string name,
								 OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = log10(v" + to_string(compiler_state.variable_index + 1) + ");\n");
	return 0;
}
std::string Log10Impl::generate_ocl_eager(FType res_type,
										  std::vector<FType> parameter_types) {
	const string conv = parameter_types[0] == F_INT32	? "(float)"
						: parameter_types[0] == F_INT64 ? "(double)"
														: "";
	return "if(index >= num_entriesR) return;\n"
		   "R[index] = "
		   "log10(" +
		   conv + "P0[p0_is_constant ? 0 : index]);";
}
void Log10Impl::execute_cpu(const FGraphNode *node,
							vector<CPUResultData> predecessor_data,
							void *__restrict__ result, size_t from,
							size_t size) {
	UNARY_EXECUTE_IMPL
}

template <typename A>
void SignImpl::unary_expression(int *__restrict result,
								const A *__restrict data1, size_t from,
								size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] =
			data1[curr->predecessors[0]->operation.op_type == FGEN_CONSTANT
					  ? 0
					  : i] < 0
				? -1
				: 1;
}
int SignImpl::generate_ocl_lazy(const FGraphNode *node, string name,
								OCLLazyCodegenState &compiler_state) {

	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = v" + to_string(compiler_state.variable_index + 1) +
		" < 0 ? -1 : 1;\n");
	return 0;
}
std::string SignImpl::generate_ocl_eager(FType res_type,
										 std::vector<FType> parameter_types) {
	return "if(index >= num_entriesR) return;\n"
		   "R[index] = "
		   "P0[p0_is_constant ? 0 : index] >= 0 ? 1 : -1;";
}
void SignImpl::execute_cpu(const FGraphNode *node,
						   vector<CPUResultData> predecessor_data,
						   void *__restrict__ result, size_t from,
						   size_t size) {
	DISPATCH_UNARY_OPERATION(int)
}
void EvenImpl::execute_cpu(const FGraphNode *node,
						   vector<CPUResultData> predecessor_data,
						   void *__restrict__ result, size_t from,
						   size_t size) {
	for (size_t i = from; i < from + size; i++)
		switch (predecessor_data[0].type) {
		case F_INT32:
			((int *__restrict__)result)[i] =
				((int *__restrict__)predecessor_data[0]
					 .data)[node->predecessors[0]->operation.op_type ==
									FGEN_CONSTANT
								? 0
								: i] %
						2
					? 0
					: 1;
			break;
		case F_INT64:
			((int *__restrict__)result)[i] =
				((long *__restrict__)predecessor_data[0]
					 .data)[node->predecessors[0]->operation.op_type ==
									FGEN_CONSTANT
								? 0
								: i] %
						2
					? 0
					: 1;
			break;
		case F_FLOAT32:
			((int *__restrict__)
				 result)[node->predecessors[0]->operation.op_type ==
								 FGEN_CONSTANT
							 ? 0
							 : i] = 0;
			break;
		case F_FLOAT64:
			((int *__restrict__)
				 result)[node->predecessors[0]->operation.op_type ==
								 FGEN_CONSTANT
							 ? 0
							 : i] = 0;
			break;
		}
}
int EvenImpl::generate_ocl_lazy(const FGraphNode *node, string name,
								OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = v" + to_string(compiler_state.variable_index + 1) +
		" % 2 == 0 ? 1 : 0;\n");
	return 0;
}
std::string EvenImpl::generate_ocl_eager(FType res_type,
										 std::vector<FType> parameter_types) {
	return "if(index >= num_entriesR) return;\n"
		   "R[index] = "
		   "P0[p0_is_constant ? 0 : index] % 2 == 0 ? 1 : 0;";
}
FGraphNode *SinImpl::local_gradient(FGraphNode *y, int dx_i,
									FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	if (0 == dx_i) {
		return fmul(prev_adj, fcos(a));
	} else
		return nullptr;
}
template <typename T, typename A>
void SinImpl::unary_expression(T *__restrict__ result,
							   const A *__restrict__ data, size_t from,
							   size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] =
			sin(data[curr->predecessors[0]->operation.op_type == FGEN_CONSTANT
						 ? 0
						 : i]);
}
int SinImpl::generate_ocl_lazy(const FGraphNode *node, string name,
							   OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = sin(v" + to_string(compiler_state.variable_index + 1) + ");\n");
	return 0;
}
std::string SinImpl::generate_ocl_eager(FType res_type,
										std::vector<FType> parameter_types) {
	return "if(index >= num_entriesR) return;\n"
		   "R[index] = "
		   "sin(P0[p0_is_constant ? 0 : index]);";
}
void SinImpl::execute_cpu(const FGraphNode *node,
						  vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from,
						  size_t size){UNARY_EXECUTE_IMPL}

FGraphNode *CosImpl::local_gradient(FGraphNode *y, int dx_i,
									FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	if (0 == dx_i) {
		return fmul(prev_adj, fneg(fsin(a)));
	} else
		return nullptr;
}
template <typename T, typename A>
void CosImpl::unary_expression(T *__restrict__ result,
							   const A *__restrict__ data, size_t from,
							   size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] =
			cos(data[curr->predecessors[0]->operation.op_type == FGEN_CONSTANT
						 ? 0
						 : i]);
}
int CosImpl::generate_ocl_lazy(const FGraphNode *node, string name,
							   OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = cos(v" + to_string(compiler_state.variable_index + 1) + ");\n");
	return 0;
}
std::string CosImpl::generate_ocl_eager(FType res_type,
										std::vector<FType> parameter_types) {
	return "if(index >= num_entriesR) return;\n"
		   "R[index] = "
		   "cos(P0[p0_is_constant ? 0 : index]);";
}
void CosImpl::execute_cpu(const FGraphNode *node,
						  vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from,
						  size_t size){UNARY_EXECUTE_IMPL}

FGraphNode *TanImpl::local_gradient(FGraphNode *y, int dx_i,
									FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	if (0 == dx_i) {
		return fmul(prev_adj, fpow(fcos(a), -2));
	} else
		return nullptr;
}
template <typename T, typename A>
void TanImpl::unary_expression(T *__restrict__ result,
							   const A *__restrict__ data, size_t from,
							   size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] =
			tan(data[curr->predecessors[0]->operation.op_type == FGEN_CONSTANT
						 ? 0
						 : i]);
}
int TanImpl::generate_ocl_lazy(const FGraphNode *node, string name,
							   OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = tan(v" + to_string(compiler_state.variable_index + 1) + ");\n");
	return 0;
}
std::string TanImpl::generate_ocl_eager(FType res_type,
										std::vector<FType> parameter_types) {
	return "if(index >= num_entriesR) return;\n"
		   "R[index] = "
		   "tan(P0[p0_is_constant ? 0 : index]);";
}
void TanImpl::execute_cpu(const FGraphNode *node,
						  vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from,
						  size_t size){UNARY_EXECUTE_IMPL}

FGraphNode *ASinImpl::local_gradient(FGraphNode *y, int dx_i,
									 FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	if (0 == dx_i) {
		return fdiv(prev_adj, fsqrt_g(fsub_icd(1.0, fmul(a, a))));
	} else
		return nullptr;
}
template <typename T, typename A>
void ASinImpl::unary_expression(T *__restrict__ result,
								const A *__restrict__ data, size_t from,
								size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] =
			asin(data[curr->predecessors[0]->operation.op_type == FGEN_CONSTANT
						  ? 0
						  : i]);
}
int ASinImpl::generate_ocl_lazy(const FGraphNode *node, string name,
								OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = asin(v" + to_string(compiler_state.variable_index + 1) + ");\n");
	return 0;
}
std::string ASinImpl::generate_ocl_eager(FType res_type,
										 std::vector<FType> parameter_types) {
	return "if(index >= num_entriesR) return;\n"
		   "R[index] = "
		   "asin(P0[p0_is_constant ? 0 : index]);";
}
void ASinImpl::execute_cpu(const FGraphNode *node,
						   vector<CPUResultData> predecessor_data,
						   void *__restrict__ result, size_t from,
						   size_t size){UNARY_EXECUTE_IMPL}

FGraphNode *ACosImpl::local_gradient(FGraphNode *y, int dx_i,
									 FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	if (0 == dx_i) {
		return fdiv(prev_adj, fneg(fsqrt_g(fsub_icd(1.0, fmul(a, a)))));
	} else
		return nullptr;
}
template <typename T, typename A>
void ACosImpl::unary_expression(T *__restrict__ result,
								const A *__restrict__ data, size_t from,
								size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] =
			acos(data[curr->predecessors[0]->operation.op_type == FGEN_CONSTANT
						  ? 0
						  : i]);
}
int ACosImpl::generate_ocl_lazy(const FGraphNode *node, string name,
								OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = acos(v" + to_string(compiler_state.variable_index + 1) + ");\n");
	return 0;
}
std::string ACosImpl::generate_ocl_eager(FType res_type,
										 std::vector<FType> parameter_types) {
	return "if(index >= num_entriesR) return;\n"
		   "R[index] = "
		   "acos(P0[p0_is_constant ? 0 : index]);";
}
void ACosImpl::execute_cpu(const FGraphNode *node,
						   vector<CPUResultData> predecessor_data,
						   void *__restrict__ result, size_t from,
						   size_t size){UNARY_EXECUTE_IMPL}

FGraphNode *ATanImpl::local_gradient(FGraphNode *y, int dx_i,
									 FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	if (0 == dx_i) {
		return fdiv(prev_adj, fadd_ci(fmul(a, a), 1));
	} else
		return nullptr;
}
template <typename T, typename A>
void ATanImpl::unary_expression(T *__restrict__ result,
								const A *__restrict__ data, size_t from,
								size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] =
			atan(data[curr->predecessors[0]->operation.op_type == FGEN_CONSTANT
						  ? 0
						  : i]);
}
std::string ATanImpl::generate_ocl_eager(FType res_type,
										 std::vector<FType> parameter_types) {
	return "if(index >= num_entriesR) return;\n"
		   "R[index] = "
		   "atan(P0[p0_is_constant ? 0 : index]);";
}
int ATanImpl::generate_ocl_lazy(const FGraphNode *node, string name,
								OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = atan(v" + to_string(compiler_state.variable_index + 1) + ");\n");
	return 0;
}
void ATanImpl::execute_cpu(const FGraphNode *node,
						   vector<CPUResultData> predecessor_data,
						   void *__restrict__ result, size_t from,
						   size_t size){UNARY_EXECUTE_IMPL}

FGraphNode *SqrtImpl::local_gradient(FGraphNode *y, int dx_i,
									 FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	if (0 == dx_i) {
		return fdiv(prev_adj, fmul_ci(y, 2));
	} else
		return nullptr;
}
template <typename T, typename A>
void SqrtImpl::unary_expression(T *__restrict__ result,
								const A *__restrict__ data, size_t from,
								size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] =
			sqrt(data[curr->predecessors[0]->operation.op_type == FGEN_CONSTANT
						  ? 0
						  : i]);
}
int SqrtImpl::generate_ocl_lazy(const FGraphNode *node, string name,
								OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = sqrt(v" + to_string(compiler_state.variable_index + 1) + ");\n");
	return 0;
}
std::string SqrtImpl::generate_ocl_eager(FType res_type,
										 std::vector<FType> parameter_types) {
	return "if(index >= num_entriesR) return;\n"
		   "R[index] = "
		   "sqrt(P0[p0_is_constant ? 0 : index]);";
}
void SqrtImpl::execute_cpu(const FGraphNode *node,
						   vector<CPUResultData> predecessor_data,
						   void *__restrict__ result, size_t from,
						   size_t size){UNARY_EXECUTE_IMPL}

FGraphNode *ExpImpl::local_gradient(FGraphNode *y, int dx_i,
									FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	if (0 == dx_i) {
		return fmul(prev_adj, y);
	} else
		return nullptr;
}
template <typename T, typename A>
void ExpImpl::unary_expression(T *__restrict__ result,
							   const A *__restrict__ data, size_t from,
							   size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] =
			exp(data[curr->predecessors[0]->operation.op_type == FGEN_CONSTANT
						 ? 0
						 : i]);
}
int ExpImpl::generate_ocl_lazy(const FGraphNode *node, string name,
							   OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = exp(v" + to_string(compiler_state.variable_index + 1) + ");\n");
	return 0;
}
std::string ExpImpl::generate_ocl_eager(FType res_type,
										std::vector<FType> parameter_types) {
	return "if(index >= num_entriesR) return;\n"
		   "R[index] = "
		   "exp(P0[p0_is_constant ? 0 : index]);";
}
void ExpImpl::execute_cpu(const FGraphNode *node,
						  vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from,
						  size_t size){UNARY_EXECUTE_IMPL}

FGraphNode *AbsImpl::local_gradient(FGraphNode *y, int dx_i,
									FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	if (0 == dx_i) {
		return fmul(prev_adj, fsub(fsign(a), fequal(a, 0.0)));
	} else
		return nullptr;
}
template <typename T>
void AbsImpl::unary_expression(T *__restrict__ result,
							   const T *__restrict__ data, size_t from,
							   size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] =
			abs(data[curr->predecessors[0]->operation.op_type == FGEN_CONSTANT
						 ? 0
						 : i]);
}
int AbsImpl::generate_ocl_lazy(const FGraphNode *node, string name,
							   OCLLazyCodegenState &compiler_state) {
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = v" + to_string(compiler_state.variable_index + 1) + " < 0 ? -v" +
		to_string(compiler_state.variable_index + 1) + ": v" +
		to_string(compiler_state.variable_index + 1) + ";\n");
	return 0;
}
std::string AbsImpl::generate_ocl_eager(FType res_type,
										std::vector<FType> parameter_types) {
	return "if(index >= num_entriesR) return;\n"
		   "R[index] = "
		   "P0[p0_is_constant ? 0 : index] < 0 ? -P0[p0_is_constant ? 0 : "
		   "index] : P0[p0_is_constant ? 0 : index];";
}
void AbsImpl::execute_cpu(const FGraphNode *node,
						  vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
