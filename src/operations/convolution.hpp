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
#ifndef FLINT_CONVOLUTION_HPP
#define FLINT_CONVOLUTION_HPP
#include "implementation.hpp"
#include "../backend_ocl/utils.hpp"

struct ConvolveImpl : OperationImplementation {
		template <typename T, typename A, typename B>
		static void binary_expression(T *__restrict__ result,
									  const A *__restrict__ data1,
									  const B *__restrict__ data2, size_t from,
									  size_t size, size_t index_man_1,
									  size_t inv_man_1, size_t index_man_2,
									  size_t inv_man_2, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override;
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override;
		void push_parameter_kernel_parameters(FGraphNode *node, FGraphNode *pred,
									   cl_kernel kernel, cl_context context,
									   int &par_index,
									   std::list<cl_mem> &to_free) override {
			push_per_parameter_dimension(pred->operation, kernel, par_index);
		}
};

struct GradientConvolve1Impl : OperationImplementation {
		template <typename T, typename A, typename B>
		static void binary_expression(T *__restrict__ result,
									  const A *__restrict__ data1,
									  const B *__restrict__ data2, size_t from,
									  size_t size, size_t index_man_1,
									  size_t inv_man_1, size_t index_man_2,
									  size_t inv_man_2, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override;
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override;
		void push_parameter_kernel_parameters(FGraphNode *node, FGraphNode *pred,
									   cl_kernel kernel, cl_context context,
									   int &par_index,
									   std::list<cl_mem> &to_free) override {
			push_per_parameter_dimension(pred->operation, kernel, par_index);
		}
};
struct GradientConvolve2Impl : OperationImplementation {
		template <typename T, typename A, typename B>
		static void binary_expression(T *__restrict__ result,
									  const A *__restrict__ data1,
									  const B *__restrict__ data2, size_t from,
									  size_t size, size_t index_man_1,
									  size_t inv_man_1, size_t index_man_2,
									  size_t inv_man_2, const FGraphNode *curr);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		int generate_ocl_lazy(const FGraphNode *node, std::string name,
							  OCLLazyCodegenState &compiler_state) override;
		std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) override;
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override;
		void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override;
		void push_parameter_kernel_parameters(FGraphNode *node, FGraphNode *pred,
									   cl_kernel kernel, cl_context context,
									   int &par_index,
									   std::list<cl_mem> &to_free) override {
			push_per_parameter_dimension(pred->operation, kernel, par_index);
		}
};
#endif
