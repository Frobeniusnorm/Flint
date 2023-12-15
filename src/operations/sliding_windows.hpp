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
#ifndef FLINT_SLIDING_WINDOWS_HPP
#define FLINT_SLIDING_WINDOWS_HPP
#include "implementation.hpp"

struct SlidingWindowImpl : OperationImplementation {
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
			return node->operation.shape[0];
		}
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		void free_additional_data(FGraphNode *gn) override {
			FSlidingWindow *s = (FSlidingWindow *)gn->operation.additional_data;
			free(s->step);
			free(s->size);
			delete s;
		}
};
struct UnslideWindowImpl : OperationImplementation {
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
			return node->predecessors[0]->operation.shape[0];
		}
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		void free_additional_data(FGraphNode *gn) override {
			free(gn->operation.additional_data);
		}
};

#endif
