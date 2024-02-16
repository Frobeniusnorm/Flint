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
#include "../backend_ocl/utils.hpp"
#include "implementation.hpp"

struct ConvolveImpl : OperationImplementation {
		/**
		 * Slides windows with the size of kernel along the shape of a and
		 * accumulates for each element of the kernel the values of the a and
		 * prev_adj that are slid against it. Additionally reprojects the values
		 * of the adjoint gradient of the convolution operation to the position
		 * where each value was calculated in a and multiplies it with the
		 * corresponding elements of the kernel before they are accumulated.
		 * Finally this yield the gradient of a.
		 */
		static FGraphNode *gradient_convolve2(FGraphNode *a, FGraphNode *kernel,
											  FGraphNode *prev_adj,
											  const unsigned int *steps);
		/**
		 * Slides kernel along the shape of a and accumulates for each element
		 * of a the values of the kernel that are slid against it. Additionally
		 * reprojects the values of the adjoint gradient of the convolution
		 * operation to the position where each value was calculated in a and
		 * multiplies it with the corresponding elements of the kernel before
		 * they are accumulated. Finally this yield the gradient of a.
		 */
		static FGraphNode *gradient_convolve1(FGraphNode *a, FGraphNode *kernel,
											  FGraphNode *prev_adj,
											  const unsigned int *steps);
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
		void
		push_parameter_kernel_parameters(FGraphNode *node, FGraphNode *pred,
										 cl_kernel kernel, cl_context context,
										 int &par_index,
										 std::list<cl_mem> &to_free) override {
			push_per_parameter_dimension(pred->operation, kernel, par_index);
		}
		int operation_score(FGraphNode *node) override {
			size_t no_elems = 1;
			const FGraphNode *a = node->predecessors[1];
			for (int i = 0; i < a->operation.dimensions; i++) {
				no_elems *= a->operation.shape[i];
			}
			return 5 + (int)(sqrt(no_elems));
		}
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		void free_additional_data(FGraphNode *gn) override {
			free(gn->operation.additional_data);
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
		void
		push_parameter_kernel_parameters(FGraphNode *node, FGraphNode *pred,
										 cl_kernel kernel, cl_context context,
										 int &par_index,
										 std::list<cl_mem> &to_free) override {
			push_per_parameter_dimension(pred->operation, kernel, par_index);
		}
		int operation_score(FGraphNode *node) override {
			size_t no_elems = 1;
			const FGraphNode *a = node->predecessors[1];
			for (int i = 0; i < a->operation.dimensions; i++) {
				no_elems *= a->operation.shape[i];
			}
			return 5 + (int)(sqrt(no_elems));
		}
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		void free_additional_data(FGraphNode *gn) override {
			free(gn->operation.additional_data);
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
		void
		push_parameter_kernel_parameters(FGraphNode *node, FGraphNode *pred,
										 cl_kernel kernel, cl_context context,
										 int &par_index,
										 std::list<cl_mem> &to_free) override {
			push_per_parameter_dimension(pred->operation, kernel, par_index);
		}
		int operation_score(FGraphNode *node) override { return 10; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		void free_additional_data(FGraphNode *gn) override {
			free(gn->operation.additional_data);
		}
		size_t deploy_as_many_elements(const FGraphNode *node) override;
};
#endif
