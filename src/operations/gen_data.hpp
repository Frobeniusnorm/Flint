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
#ifndef FLINT_GEN_DATA_HPP
#define FLINT_GEN_DATA_HPP
#include "implementation.hpp"

struct GenRandomImpl : OperationImplementation {
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
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override {
			return nullptr;
		}
		void free_additional_data(FGraphNode *gn) override {
			free(gn->operation.additional_data);
		}
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_FLOAT64}};
		}
};

struct GenConstantImpl : OperationImplementation {
		template <typename T>
		void zeroary_expression(const FGraphNode *node, T *__restrict__ result,
								size_t from, size_t size);
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
		int operation_score(FGraphNode *node) override { return 10; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override {
			return nullptr;
		}
		void free_additional_data(FGraphNode *gn) override {
			free(gn->operation.additional_data);
		}
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_INT32}, {F_INT64}, {F_FLOAT32}, {F_FLOAT64}};
		}
};

struct GenArangeImpl : OperationImplementation {
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
		int operation_score(FGraphNode *node) override { return 5; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override {
			return nullptr;
		}
		void free_additional_data(FGraphNode *gn) override {
			free(gn->operation.additional_data);
		}
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_INT64}};
		}
};
#endif
