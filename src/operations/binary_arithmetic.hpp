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
#ifndef FLINT_BINARY_ARITHMETIC_HPP
#define FLINT_BINARY_ARITHMETIC_HPP
#include "../backend_ocl/utils.hpp"
#include "implementation.hpp"
struct AddImpl : OperationImplementation {
		static std::vector<bool>
		reuse_parameter_binary_impl(const FGraphNode *node);
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
		virtual int
		generate_ocl_lazy(const FGraphNode *node, std::string name,
						  OCLLazyCodegenState &compiler_state) override;
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return reuse_parameter_binary_impl(node);
		}
};
struct SubImpl : OperationImplementation {
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
		virtual int
		generate_ocl_lazy(const FGraphNode *node, std::string name,
						  OCLLazyCodegenState &compiler_state) override;
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return AddImpl::reuse_parameter_binary_impl(node);
		}
};
struct MulImpl : OperationImplementation {
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
		virtual int
		generate_ocl_lazy(const FGraphNode *node, std::string name,
						  OCLLazyCodegenState &compiler_state) override;
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return AddImpl::reuse_parameter_binary_impl(node);
		}
};
struct DivImpl : OperationImplementation {
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
		virtual int
		generate_ocl_lazy(const FGraphNode *node, std::string name,
						  OCLLazyCodegenState &compiler_state) override;
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return AddImpl::reuse_parameter_binary_impl(node);
		}
};
struct PowImpl : OperationImplementation {
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
		virtual int
		generate_ocl_lazy(const FGraphNode *node, std::string name,
						  OCLLazyCodegenState &compiler_state) override;
		int operation_score(FGraphNode *node) override { return 1; }
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
		std::vector<bool>
		reuse_parameter_result(const FGraphNode *node) override {
			return AddImpl::reuse_parameter_binary_impl(node);
		}
};
struct MatMulImpl : OperationImplementation {
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
		virtual int
		generate_ocl_lazy(const FGraphNode *node, std::string name,
						  OCLLazyCodegenState &compiler_state) override;
		int operation_score(FGraphNode *node) override {
			const FGraphNode *a = node->predecessors[0];
			return 5 * a->operation.shape[a->operation.dimensions - 1];
		}
		FGraphNode *local_gradient(FGraphNode *y, int dx_i,
								   FGraphNode *prev_adj) override;
};
#endif
