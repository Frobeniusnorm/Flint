#include "binary_arithmetic.hpp"

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
template <typename T, typename A, typename B>
void MatMulImpl::binary_expression(T *__restrict__ result,
								   const A *__restrict__ data1,
								   const B *__restrict__ data2, size_t from,
								   size_t size, size_t index_man_1,
								   size_t inv_man_1, size_t index_man_2,
								   size_t inv_man_2, const FGraphNode *curr) {
	FGraphNode *gnp1 = curr->predecessors[0], *gnp2 = curr->predecessors[1];
	size_t l = gnp1->operation.shape[gnp1->operation.dimensions - 2];
	size_t m = gnp1->operation.shape[gnp1->operation.dimensions - 1];
	size_t n = gnp2->operation.shape[gnp2->operation.dimensions - 1];
	for (size_t index = from; index < from + size; index++) {
		result[index] = 0;
		// indices in node matrix
		size_t j = (index % (l * n)) / n;
		size_t k = (index % (l * n)) % n;

		// matrix number of predecessors
		size_t base_p1 = 0;
		if (gnp1->operation.dimensions > 2) {
			// get matrix number of index and then reproject
			base_p1 = (index / (l * n)) * (l * m);
		}
		size_t base_p2 = 0;
		if (gnp2->operation.dimensions > 2) {
			// get matrix number of index and then reproject
			base_p2 = (index / (l * n)) * (m * n);
		}
		for (size_t i = 0; i < m; i++) {
			result[index] +=
				data1[base_p1 + j * m + i] * data2[base_p2 + i * n + k];
		}
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
void MatMulImpl::execute_cpu(const FGraphNode *node,
							 std::vector<CPUResultData> predecessor_data,
							 void *__restrict__ result, size_t from,
							 size_t size) {
	BINARY_EXECUTE_IMPL
}
