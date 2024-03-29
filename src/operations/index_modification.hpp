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
#ifndef FLINT_INDEX_MODIFICATION_HPP
#define FLINT_INDEX_MODIFICATION_HPP
#include "../backend_ocl/utils.hpp"
#include "flint.h"
#include "implementation.hpp"
#include "binary_arithmetic.hpp"

struct SliceImpl : OperationImplementation {
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
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override;
		void
		push_parameter_kernel_parameters(FGraphNode *node, FGraphNode *pred,
										 cl_kernel kernel, cl_context context,
										 int &par_index,
										 std::list<cl_mem> &to_free) override;
		int operation_score(FGraphNode *node) override {
			if (!fIsEagerExecution()) {
				size_t sliced_away = 1;
				const FGraphNode *p = node->predecessors[0];
				for (int i = 0; i < node->operation.dimensions; i++) {
					long diff =
						(p->operation.shape[i] - node->operation.shape[i]);
					if (diff > 0)
						sliced_away *= diff;
				}
				return sliced_away;
			} else
				return 3;
		}
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		void free_additional_data(FGraphNode *gn) override {
			FSlice *s = (FSlice *)gn->operation.additional_data;
			free(s->end);
			free(s->start);
			free(s->step);
			delete s;
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return AddImpl::reuse_parameter_binary_impl(node);
		}
};
struct ExtendImpl : OperationImplementation {
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
		std::string generate_ocl_parameters_eager(
			FType res_type, std::vector<FType> parameter_types) override;
		void
		push_parameter_kernel_parameters(FGraphNode *node, FGraphNode *pred,
										 cl_kernel kernel, cl_context context,
										 int &par_index,
										 std::list<cl_mem> &to_free) override;
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		void free_additional_data(FGraphNode *gn) override {
			FExtend *s = (FExtend *)gn->operation.additional_data;
			free(s->start);
			free(s->step);
      delete s;
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return AddImpl::reuse_parameter_binary_impl(node);
		}
};
struct IndexImpl : OperationImplementation {
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
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override;
};
struct SetIndexImpl : OperationImplementation {
		template <typename T>
		void execute_cpu_typed(const FGraphNode *node,
							   std::vector<CPUResultData> predecessor_data,
							   T *__restrict__ result, size_t from,
							   size_t size);
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
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override;
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {true, false, false};
		}
};
#endif
