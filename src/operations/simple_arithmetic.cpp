#include "simple_arithmetic.hpp"

template <typename T, typename A, typename B>
void AddImpl::binary_expression(T *__restrict__ result,
							   const A *__restrict__ data1,
							   const B *__restrict__ data2, size_t from,
							   size_t size, size_t index_man_1,
							   size_t inv_man_1, size_t index_man_2,
							   size_t inv_man_2, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++) {
		result[i] = data1[(i / inv_man_1) % index_man_1] +
					data2[(i / inv_man_2) % index_man_2];
	}
}
template <typename T, typename A, typename B>
void SubImpl::binary_expression(T *__restrict__ result,
							   const A *__restrict__ data1,
							   const B *__restrict__ data2, size_t from,
							   size_t size, size_t index_man_1,
							   size_t inv_man_1, size_t index_man_2,
							   size_t inv_man_2, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++) {
		result[i] = data1[(i / inv_man_1) % index_man_1] -
					data2[(i / inv_man_2) % index_man_2];
	}
}
template <typename T, typename A, typename B>
void MulImpl::binary_expression(T *__restrict__ result,
							   const A *__restrict__ data1,
							   const B *__restrict__ data2, size_t from,
							   size_t size, size_t index_man_1,
							   size_t inv_man_1, size_t index_man_2,
							   size_t inv_man_2, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++) {
		result[i] = data1[(i / inv_man_1) % index_man_1] *
					data2[(i / inv_man_2) % index_man_2];
	}
}
template <typename T, typename A, typename B>
void DivImpl::binary_expression(T *__restrict__ result,
							   const A *__restrict__ data1,
							   const B *__restrict__ data2, size_t from,
							   size_t size, size_t index_man_1,
							   size_t inv_man_1, size_t index_man_2,
							   size_t inv_man_2, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++) {
		result[i] = data1[(i / inv_man_1) % index_man_1] /
					data2[(i / inv_man_2) % index_man_2];
	}
}
template <typename T, typename A, typename B>
void PowImpl::binary_expression(T *__restrict__ result,
							   const A *__restrict__ data1,
							   const B *__restrict__ data2, size_t from,
							   size_t size, size_t index_man_1,
							   size_t inv_man_1, size_t index_man_2,
							   size_t inv_man_2, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++) {
		result[i] = pow(data1[(i / inv_man_1) % index_man_1],
						data2[(i / inv_man_2) % index_man_2]);
	}
}
void SubImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	BINARY_EXECUTE_IMPL
}
void AddImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	BINARY_EXECUTE_IMPL
}
void MulImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	BINARY_EXECUTE_IMPL
}
void DivImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	BINARY_EXECUTE_IMPL
}
void PowImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	BINARY_EXECUTE_IMPL
}
