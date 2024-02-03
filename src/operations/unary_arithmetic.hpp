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
#ifndef FLINT_UNARY_ARITHMETIC_HPP
#define FLINT_UNARY_ARITHMETIC_HPP
#include "flint.h"
#include "implementation.hpp"
#include "src/utils.hpp"

static std::string
unary_impl_generate_ocl_parameters_eager(FType res_type,
										 std::vector<FType> parameter_types) {
	return ", const __global " + type_string(parameter_types[0]) +
		   "* P0"
		   ", const long num_entries0, const int p0_is_constant";
}
static void
unary_impl_push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
											 cl_context context, int &par_index,
											 std::list<cl_mem> &to_free) {
	const int is_constant =
		node->predecessors[0]->operation.op_type == FGEN_CONSTANT ? 1 : 0;
	if (clSetKernelArg(kernel, par_index++, sizeof(int),
					   (void *)&is_constant) != CL_SUCCESS) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Could not load Argument to kernel!");
		return;
	}
}

struct NegImpl : OperationImplementation {
		template <typename T>
		static void unary_expression(T *__restrict__ result,
									 const T *__restrict__ data1, size_t from,
									 size_t size, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		virtual int operation_score(FGraphNode *node) override { return 1; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {true};
		}
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override {
			return unary_impl_generate_ocl_parameters_eager(res_type,
															parameter_types);
		}
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override {
			unary_impl_push_additional_kernel_parameters(node, kernel, context,
														 par_index, to_free);
		}
};
struct LogImpl : OperationImplementation {
		template <typename T, typename A>
		static void unary_expression(T *__restrict__ result,
									 const A *__restrict__ data1, size_t from,
									 size_t size, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		virtual int operation_score(FGraphNode *node) override { return 3; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_FLOAT32, F_FLOAT32}, {F_FLOAT64, F_FLOAT64}};
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {true};
		}
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override {
			return unary_impl_generate_ocl_parameters_eager(res_type,
															parameter_types);
		}
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override {
			unary_impl_push_additional_kernel_parameters(node, kernel, context,
														 par_index, to_free);
		}
};
struct Log2Impl : OperationImplementation {
		template <typename T, typename A>
		static void unary_expression(T *__restrict__ result,
									 const A *__restrict__ data1, size_t from,
									 size_t size, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		virtual int operation_score(FGraphNode *node) override { return 3; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_FLOAT32, F_FLOAT32}, {F_FLOAT64, F_FLOAT64}};
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {true};
		}
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override {
			return unary_impl_generate_ocl_parameters_eager(res_type,
															parameter_types);
		}
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override {
			unary_impl_push_additional_kernel_parameters(node, kernel, context,
														 par_index, to_free);
		}
};
struct Log10Impl : OperationImplementation {
		template <typename T, typename A>
		static void unary_expression(T *__restrict__ result,
									 const A *__restrict__ data1, size_t from,
									 size_t size, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		virtual int operation_score(FGraphNode *node) override { return 3; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_FLOAT32, F_FLOAT32}, {F_FLOAT64, F_FLOAT64}};
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {true};
		}
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override {
			return unary_impl_generate_ocl_parameters_eager(res_type,
															parameter_types);
		}
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override {
			unary_impl_push_additional_kernel_parameters(node, kernel, context,
														 par_index, to_free);
		}
};
struct SignImpl : OperationImplementation {
		template <typename A>
		static void unary_expression(int *__restrict__ result,
									 const A *__restrict__ data1, size_t from,
									 size_t size, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		int operation_score(FGraphNode *node) override { return 1; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override {
			return nullptr;
		}
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_INT32, F_INT32},
					{F_INT32, F_INT64},
					{F_INT32, F_FLOAT32},
					{F_INT32, F_FLOAT64}};
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {type_size(node->predecessors[0]->operation.data_type) ==
					type_size(F_INT32)};
		}
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override {
			return unary_impl_generate_ocl_parameters_eager(res_type,
															parameter_types);
		}
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override {
			unary_impl_push_additional_kernel_parameters(node, kernel, context,
														 par_index, to_free);
		}
};
struct EvenImpl : OperationImplementation {
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		virtual int operation_score(FGraphNode *node) override { return 1; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override {
			return nullptr;
		}
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_INT32, F_INT32}, {F_INT32, F_INT64}};
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {type_size(node->predecessors[0]->operation.data_type) ==
					type_size(F_INT32)};
		}
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override {
			return unary_impl_generate_ocl_parameters_eager(res_type,
															parameter_types);
		}
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override {
			unary_impl_push_additional_kernel_parameters(node, kernel, context,
														 par_index, to_free);
		}
};
struct SinImpl : OperationImplementation {
		template <typename T, typename A>
		static void unary_expression(T *__restrict__ result,
									 const A *__restrict__ data1, size_t from,
									 size_t size, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		virtual int operation_score(FGraphNode *node) override { return 3; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_FLOAT32, F_FLOAT32}, {F_FLOAT64, F_FLOAT64}};
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {true};
		}
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override {
			return unary_impl_generate_ocl_parameters_eager(res_type,
															parameter_types);
		}
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override {
			unary_impl_push_additional_kernel_parameters(node, kernel, context,
														 par_index, to_free);
		}
};
struct CosImpl : OperationImplementation {
		template <typename T, typename A>
		static void unary_expression(T *__restrict__ result,
									 const A *__restrict__ data1, size_t from,
									 size_t size, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		virtual int operation_score(FGraphNode *node) override { return 3; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_FLOAT32, F_FLOAT32}, {F_FLOAT64, F_FLOAT64}};
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {true};
		}
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override {
			return unary_impl_generate_ocl_parameters_eager(res_type,
															parameter_types);
		}
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override {
			unary_impl_push_additional_kernel_parameters(node, kernel, context,
														 par_index, to_free);
		}
};
struct TanImpl : OperationImplementation {
		template <typename T, typename A>
		static void unary_expression(T *__restrict__ result,
									 const A *__restrict__ data1, size_t from,
									 size_t size, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		virtual int operation_score(FGraphNode *node) override { return 3; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_FLOAT32, F_FLOAT32}, {F_FLOAT64, F_FLOAT64}};
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {true};
		}
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override {
			return unary_impl_generate_ocl_parameters_eager(res_type,
															parameter_types);
		}
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override {
			unary_impl_push_additional_kernel_parameters(node, kernel, context,
														 par_index, to_free);
		}
};
struct ASinImpl : OperationImplementation {
		template <typename T, typename A>
		static void unary_expression(T *__restrict__ result,
									 const A *__restrict__ data1, size_t from,
									 size_t size, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		virtual int operation_score(FGraphNode *node) override { return 3; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_FLOAT32, F_FLOAT32}, {F_FLOAT64, F_FLOAT64}};
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {true};
		}
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override {
			return unary_impl_generate_ocl_parameters_eager(res_type,
															parameter_types);
		}
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override {
			unary_impl_push_additional_kernel_parameters(node, kernel, context,
														 par_index, to_free);
		}
};
struct ACosImpl : OperationImplementation {
		template <typename T, typename A>
		static void unary_expression(T *__restrict__ result,
									 const A *__restrict__ data1, size_t from,
									 size_t size, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		virtual int operation_score(FGraphNode *node) override { return 3; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_FLOAT32, F_FLOAT32}, {F_FLOAT64, F_FLOAT64}};
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {true};
		}
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override {
			return unary_impl_generate_ocl_parameters_eager(res_type,
															parameter_types);
		}
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override {
			unary_impl_push_additional_kernel_parameters(node, kernel, context,
														 par_index, to_free);
		}
};
struct ATanImpl : OperationImplementation {
		template <typename T, typename A>
		static void unary_expression(T *__restrict__ result,
									 const A *__restrict__ data1, size_t from,
									 size_t size, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		virtual int operation_score(FGraphNode *node) override { return 3; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_FLOAT32, F_FLOAT32}, {F_FLOAT64, F_FLOAT64}};
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {true};
		}
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override {
			return unary_impl_generate_ocl_parameters_eager(res_type,
															parameter_types);
		}
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override {
			unary_impl_push_additional_kernel_parameters(node, kernel, context,
														 par_index, to_free);
		}
};
struct SqrtImpl : OperationImplementation {
		template <typename T, typename A>
		static void unary_expression(T *__restrict__ result,
									 const A *__restrict__ data1, size_t from,
									 size_t size, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		virtual int operation_score(FGraphNode *node) override { return 3; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_FLOAT32, F_FLOAT32}, {F_FLOAT64, F_FLOAT64}};
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {true};
		}
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override {
			return unary_impl_generate_ocl_parameters_eager(res_type,
															parameter_types);
		}
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override {
			unary_impl_push_additional_kernel_parameters(node, kernel, context,
														 par_index, to_free);
		}
};
struct ExpImpl : OperationImplementation {
		template <typename T, typename A>
		static void unary_expression(T *__restrict__ result,
									 const A *__restrict__ data1, size_t from,
									 size_t size, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		virtual int operation_score(FGraphNode *node) override { return 3; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_FLOAT32, F_FLOAT32}, {F_FLOAT64, F_FLOAT64}};
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {true};
		}
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override {
			return unary_impl_generate_ocl_parameters_eager(res_type,
															parameter_types);
		}
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override {
			unary_impl_push_additional_kernel_parameters(node, kernel, context,
														 par_index, to_free);
		}
};
struct AbsImpl : OperationImplementation {
		template <typename T>
		static void unary_expression(T *__restrict__ result,
									 const T *__restrict__ data1, size_t from,
									 size_t size, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		virtual int operation_score(FGraphNode *node) override { return 1; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {true};
		}
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override {
			return unary_impl_generate_ocl_parameters_eager(res_type,
															parameter_types);
		}
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override {
			unary_impl_push_additional_kernel_parameters(node, kernel, context,
														 par_index, to_free);
		}
};
#endif
