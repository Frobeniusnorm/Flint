#include "unary_arithmetic.hpp"
#include "src/operations/implementation.hpp"

template <typename T, typename A>
void NegImpl::unary_expression(T *__restrict__ result,
							   const A *__restrict__ data, size_t from,
							   size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = -((const T *__restrict__)data)[i];
}
void NegImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	UNARY_EXECUTE_IMPL
}
template <typename T, typename A>
void LogImpl::unary_expression(T *__restrict__ result,
							   const A *__restrict__ data, size_t from,
							   size_t size, const FGraphNode *curr) {
	for (size_t i = from; i < from + size; i++)
		result[i] = log(((const T *__restrict__)data)[i]);
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
		result[i] = log2(((const T *__restrict__)data)[i]);
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
		result[i] = log2(((const T *__restrict__)data)[i]);
}
void Log10Impl::execute_cpu(const FGraphNode *node,
							std::vector<CPUResultData> predecessor_data,
							void *__restrict__ result, size_t from,
							size_t size) {
	UNARY_EXECUTE_IMPL
}
void SignImpl::execute_cpu(const FGraphNode *node,
						   std::vector<CPUResultData> predecessor_data,
						   void *__restrict__ result, size_t from,
						   size_t size) {
	for (size_t i = from; i < from + size; i++)
		switch (predecessor_data[0].type) {
		case F_INT32:

			((int *__restrict__)result)[i] =
				((int *__restrict__)predecessor_data[0].data)[i] < 0 ? -1 : 1;
			break;
		case F_INT64:

			((int *__restrict__)result)[i] =
				((long *__restrict__)predecessor_data[0].data)[i] < 0 ? -1 : 1;
			break;
		case F_FLOAT32:

			((int *__restrict__)result)[i] =
				((float *__restrict__)predecessor_data[0].data)[i] < 0 ? -1 : 1;
			break;
		case F_FLOAT64:

			((int *__restrict__)result)[i] =
				((double *__restrict__)predecessor_data[0].data)[i] < 0 ? -1
																		: 1;
			break;
		}
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
		result[i] = sin(((const T *__restrict__)data)[i]);
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
		result[i] = cos(((const T *__restrict__)data)[i]);
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
		result[i] = tan(((const T *__restrict__)data)[i]);
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
		result[i] = asin(((const T *__restrict__)data)[i]);
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
		result[i] = acos(((const T *__restrict__)data)[i]);
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
		result[i] = atan(((const T *__restrict__)data)[i]);
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
		result[i] = sqrt(((const T *__restrict__)data)[i]);
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
		result[i] = exp(((const T *__restrict__)data)[i]);
}
void ExpImpl::execute_cpu(const FGraphNode *node,
						  std::vector<CPUResultData> predecessor_data,
						  void *__restrict__ result, size_t from, size_t size) {
	UNARY_EXECUTE_IMPL
}
