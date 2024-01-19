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
#ifndef FLINT_COMPARISON_HPP
#define FLINT_COMPARISON_HPP
#include "implementation.hpp"
#include "../utils.hpp"
#include "binary_arithmetic.hpp"

struct MinImpl : OperationImplementation {
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
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return AddImpl::reuse_parameter_binary_impl(node);
		}
};
struct MaxImpl : OperationImplementation {
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
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return AddImpl::reuse_parameter_binary_impl(node);
		}
};
struct LessImpl : OperationImplementation {
		template <typename A, typename B>
		static void binary_expression(int *__restrict__ result,
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
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			using namespace std;
			vector<vector<FType>> all_comb = all_type_permutations(2);
			for (vector<FType> &comb : all_comb)
				comb.insert(comb.begin(), F_INT32);
			return all_comb;
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {type_size(node->predecessors[0]->operation.data_type) == type_size(F_INT32)};
		}
};
struct GreaterImpl : OperationImplementation {
		template <typename A, typename B>
		static void binary_expression(int *__restrict__ result,
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
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			using namespace std;
			vector<vector<FType>> all_comb = all_type_permutations(2);
			for (vector<FType> &comb : all_comb)
				comb.insert(comb.begin(), F_INT32);
			return all_comb;
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {type_size(node->predecessors[0]->operation.data_type) == type_size(F_INT32)};
		}
};
struct EqualImpl : OperationImplementation {
		template <typename A, typename B>
		static void binary_expression(int *__restrict__ result,
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
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			using namespace std;
			vector<vector<FType>> all_comb = all_type_permutations(2);
			for (vector<FType> &comb : all_comb)
				comb.insert(comb.begin(), F_INT32);
			return all_comb;
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {type_size(node->predecessors[0]->operation.data_type) == type_size(F_INT32)};
		}
};
struct DropoutImpl : OperationImplementation {
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
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) override;
		virtual int operation_score(FGraphNode *node) override { return 2; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		void free_additional_data(FGraphNode *gn) override {
			free(gn->operation.additional_data);
		}
		std::vector<std::vector<FType>>
		kernel_type_combinations(const FGraphNode *node) override {
			return {{F_INT32, F_INT32},
					{F_INT64, F_INT64},
					{F_FLOAT32, F_FLOAT32},
					{F_FLOAT64, F_FLOAT64}};
		}
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return {true};
		}
};
#endif
