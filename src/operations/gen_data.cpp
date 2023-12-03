#include "gen_data.hpp"
#include "src/operations/implementation.hpp"
void GenRandomImpl::execute_cpu(const FGraphNode *node,
								std::vector<CPUResultData> predecessor_data,
								void *__restrict__ result, size_t from,
								size_t size) {
	double seed = ((double *)node->operation.additional_data)[0];
	for (size_t i = from; i < from + size; i++) {
		double v = sin(i + seed) * 43758.5453123;
		((double *)result)[i] = std::min(v - floor(v), 0.99999);
	}
}

template <typename T>
void GenConstantImpl::zeroary_expression(
	const FGraphNode *node,
	T *__restrict__ result, size_t from, size_t size) {
	T value = ((T *)node->operation.additional_data)[0];
	for (size_t i = from; i < from + size; i++)
		result[i] = value;
}
void GenConstantImpl::execute_cpu(const FGraphNode *node,
								  std::vector<CPUResultData> predecessor_data,
								  void *__restrict__ result, size_t from,
								  size_t size) {
	ZEROARY_EXECUTE_IMPL
}
void GenArangeImpl::execute_cpu(const FGraphNode *node,
								std::vector<CPUResultData> predecessor_data,
								void *__restrict__ result, size_t from,
								size_t size) {
	unsigned int ax = ((unsigned int *)node->operation.additional_data)[0];
	size_t acc_sizes_ax = 1;
	for (unsigned int i = ax + 1; i < node->operation.dimensions; i++)
		acc_sizes_ax *= node->operation.shape[i];
	for (size_t i = from; i < from + size; i++)
		((long *)result)[i] = (i / acc_sizes_ax) % node->operation.shape[ax];
}
