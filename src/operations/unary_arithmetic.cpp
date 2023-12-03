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
#include "src/operations/implementation.hpp"

template <typename T>
void NegImpl::unary_expression(T *__restrict__ result,
							   const T *__restrict__ data, size_t from,
							   size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = -data[i];
}
void NegImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
template <typename T, typename A>
void LogImpl::unary_expression(T *__restrict__ result,
							   const A *__restrict__ data, size_t from,
							   size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = log(data[i]);
}
void LogImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	UNARY_EXECUTE_IMPL
}
template <typename T, typename A>
void Log2Impl::unary_expression(T *__restrict__ result,
								const A *__restrict__ data, size_t from,
								size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = log2(data[i]);
}
void Log2Impl::execute_cpu(const FGraphNode *node,
						   std::vector<CPUResultData> predecessor_data,
						   void *__restrict__ result, size_t from,
						   size_t size) {
	UNARY_EXECUTE_IMPL
}
template <typename T, typename A>
void Log10Impl::unary_expression(T *__restrict__ result,
								 const A *__restrict__ data, size_t from,
								 size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = log2(data[i]);
}
void Log10Impl::execute_cpu(const FGraphNode *node,
							std::vector<CPUResultData> predecessor_data,
							void *__restrict__ result, size_t from,
							size_t size) {
	UNARY_EXECUTE_IMPL
}
template <typename A>
void SignImpl::unary_expression(int *__restrict result,
								const A *__restrict data1, size_t from,
								size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = data1[i] < 0 ? -1 : 1;
}
void SignImpl::execute_cpu(const FGraphNode *node,
						   std::vector<CPUResultData> predecessor_data,
						   void *__restrict__ result, size_t from,
						   size_t size) {
	DISPATCH_UNARY_OPERATION(int)
}
void EvenImpl::execute_cpu(const FGraphNode *node,
						   std::vector<CPUResultData> predecessor_data,
						   void *__restrict__ result, size_t from,
						   size_t size) {
	for (size_t i = from; i < from + size; i++)
		switch (predecessor_data[0].type) {
		case F_INT32:
			((int *__restrict__)result)[i] =
				((int *__restrict__)predecessor_data[0].data)[i] % 2 ? 0 : 1;
			break;
		case F_INT64:
			((int *__restrict__)result)[i] =
				((long *__restrict__)predecessor_data[0].data)[i] % 2 ? 0 : 1;
			break;
		case F_FLOAT32:
			((int *__restrict__)result)[i] = 0;
			break;
		case F_FLOAT64:
			((int *__restrict__)result)[i] = 0;
			break;
		}
}
template <typename T, typename A>
void SinImpl::unary_expression(T *__restrict__ result,
							   const A *__restrict__ data, size_t from,
							   size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = sin(data[i]);
}
void SinImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	UNARY_EXECUTE_IMPL
}
template <typename T, typename A>
void CosImpl::unary_expression(T *__restrict__ result,
							   const A *__restrict__ data, size_t from,
							   size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = cos(data[i]);
}
void CosImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	UNARY_EXECUTE_IMPL
}
template <typename T, typename A>
void TanImpl::unary_expression(T *__restrict__ result,
							   const A *__restrict__ data, size_t from,
							   size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = tan(data[i]);
}
void TanImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	UNARY_EXECUTE_IMPL
}
template <typename T, typename A>
void ASinImpl::unary_expression(T *__restrict__ result,
								const A *__restrict__ data, size_t from,
								size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = asin(data[i]);
}
void ASinImpl::execute_cpu(const FGraphNode *node,
						   std::vector<CPUResultData> predecessor_data,
						   void *__restrict__ result, size_t from,
						   size_t size) {
	UNARY_EXECUTE_IMPL
}
template <typename T, typename A>
void ACosImpl::unary_expression(T *__restrict__ result,
								const A *__restrict__ data, size_t from,
								size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = acos(data[i]);
}
void ACosImpl::execute_cpu(const FGraphNode *node,
						   std::vector<CPUResultData> predecessor_data,
						   void *__restrict__ result, size_t from,
						   size_t size) {
	UNARY_EXECUTE_IMPL
}
template <typename T, typename A>
void ATanImpl::unary_expression(T *__restrict__ result,
								const A *__restrict__ data, size_t from,
								size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = atan(data[i]);
}
void ATanImpl::execute_cpu(const FGraphNode *node,
						   std::vector<CPUResultData> predecessor_data,
						   void *__restrict__ result, size_t from,
						   size_t size) {
	UNARY_EXECUTE_IMPL
}
template <typename T, typename A>
void SqrtImpl::unary_expression(T *__restrict__ result,
								const A *__restrict__ data, size_t from,
								size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = sqrt(data[i]);
}
void SqrtImpl::execute_cpu(const FGraphNode *node,
						   std::vector<CPUResultData> predecessor_data,
						   void *__restrict__ result, size_t from,
						   size_t size) {
	UNARY_EXECUTE_IMPL
}
template <typename T, typename A>
void ExpImpl::unary_expression(T *__restrict__ result,
							   const A *__restrict__ data, size_t from,
							   size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = exp(data[i]);
}
void ExpImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	UNARY_EXECUTE_IMPL
}
template <typename T>
void AbsImpl::unary_expression(T *__restrict__ result,
							   const T *__restrict__ data, size_t from,
							   size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = abs(data[i]);
}
void AbsImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
