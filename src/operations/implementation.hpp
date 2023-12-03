#ifndef FLINT_IMPLEMENTATIONS_HPP
#define FLINT_IMPLEMENTATIONS_HPP

#include "../../flint.h"
#include "../backend_cpu/cpu_common.hpp"
#include "../utils.hpp"
#include <vector>

#define DISPATCH_BINARY_OPERATION(T)                                           \
	const CPUResultData p1 = predecessor_data[0], p2 = predecessor_data[1];    \
	size_t im1 = p1.num_entries, im2 = p2.num_entries;                         \
	size_t iv1 = 1, iv2 = 1;                                                   \
	calculateDivisorForInverseBroadcasting(node->predecessors[0], iv1,         \
										   node->predecessors[1], iv2);        \
	switch (p1.type) {                                                         \
	case F_INT32:                                                              \
		switch (p2.type) {                                                     \
		case F_INT32:                                                          \
			binary_expression((T *)result, (int *)p1.data, (int *)p2.data,     \
							  from, size, im1, iv1, im2, iv2, node);           \
			break;                                                             \
		case F_FLOAT32:                                                        \
			binary_expression((T *)result, (int *)p1.data, (float *)p2.data,   \
							  from, size, im1, iv1, im2, iv2, node);           \
			break;                                                             \
		case F_FLOAT64:                                                        \
			binary_expression((T *)result, (int *)p1.data, (double *)p2.data,  \
							  from, size, im1, iv1, im2, iv2, node);           \
			break;                                                             \
		case F_INT64:                                                          \
			binary_expression((T *)result, (int *)p1.data, (long *)p2.data,    \
							  from, size, im1, iv1, im2, iv2, node);           \
			break;                                                             \
		}                                                                      \
		break;                                                                 \
	case F_FLOAT32:                                                            \
		switch (p2.type) {                                                     \
		case F_INT32:                                                          \
			binary_expression((T *)result, (float *)p1.data, (int *)p2.data,   \
							  from, size, im1, iv1, im2, iv2, node);           \
			break;                                                             \
		case F_FLOAT32:                                                        \
			binary_expression((T *)result, (float *)p1.data, (float *)p2.data, \
							  from, size, im1, iv1, im2, iv2, node);           \
			break;                                                             \
		case F_FLOAT64:                                                        \
			binary_expression((T *)result, (float *)p1.data,                   \
							  (double *)p2.data, from, size, im1, iv1, im2,    \
							  iv2, node);                                      \
			break;                                                             \
		case F_INT64:                                                          \
			binary_expression((T *)result, (float *)p1.data, (long *)p2.data,  \
							  from, size, im1, iv1, im2, iv2, node);           \
			break;                                                             \
		}                                                                      \
		break;                                                                 \
	case F_FLOAT64:                                                            \
		switch (p2.type) {                                                     \
		case F_INT32:                                                          \
			binary_expression((T *)result, (double *)p1.data, (int *)p2.data,  \
							  from, size, im1, iv1, im2, iv2, node);           \
			break;                                                             \
		case F_FLOAT32:                                                        \
			binary_expression((T *)result, (double *)p1.data,                  \
							  (float *)p2.data, from, size, im1, iv1, im2,     \
							  iv2, node);                                      \
			break;                                                             \
		case F_FLOAT64:                                                        \
			binary_expression((T *)result, (double *)p1.data,                  \
							  (double *)p2.data, from, size, im1, iv1, im2,    \
							  iv2, node);                                      \
			break;                                                             \
		case F_INT64:                                                          \
			binary_expression((T *)result, (double *)p1.data, (long *)p2.data, \
							  from, size, im1, iv1, im2, iv2, node);           \
			break;                                                             \
		}                                                                      \
		break;                                                                 \
	case F_INT64:                                                              \
		switch (p2.type) {                                                     \
		case F_INT32:                                                          \
			binary_expression((T *)result, (long *)p1.data, (int *)p2.data,    \
							  from, size, im1, iv1, im2, iv2, node);           \
			break;                                                             \
		case F_FLOAT32:                                                        \
			binary_expression((T *)result, (long *)p1.data, (float *)p2.data,  \
							  from, size, im1, iv1, im2, iv2, node);           \
			break;                                                             \
		case F_FLOAT64:                                                        \
			binary_expression((T *)result, (long *)p1.data, (double *)p2.data, \
							  from, size, im1, iv1, im2, iv2, node);           \
			break;                                                             \
		case F_INT64:                                                          \
			binary_expression((T *)result, (long *)p1.data, (long *)p2.data,   \
							  from, size, im1, iv1, im2, iv2, node);           \
			break;                                                             \
		}                                                                      \
		break;                                                                 \
	}
/* calls a function `binary_expression` with the signature
 * template <typename T, typename A, typename B>
 * void binaryExpression(T *__restrict__ result,
 * 						const A *__restrict__ data1,
 * 						const B *__restrict__ data2, size_t from,
 * 						size_t size, size_t index_man_1,
 * 						size_t inv_man_1, size_t index_man_2,
 * 						size_t inv_man_2, const FGraphNode *curr); */
#define BINARY_EXECUTE_IMPL                                                    \
	switch (node->operation.data_type) {                                       \
	case F_INT32: {                                                            \
		DISPATCH_BINARY_OPERATION(int)                                         \
	} break;                                                                   \
	case F_FLOAT32: {                                                          \
		DISPATCH_BINARY_OPERATION(float)                                       \
	} break;                                                                   \
	case F_INT64: {                                                            \
		DISPATCH_BINARY_OPERATION(long)                                        \
	} break;                                                                   \
	case F_FLOAT64: {                                                          \
		DISPATCH_BINARY_OPERATION(double)                                      \
	} break;                                                                   \
	}
// calls a function `zeroary_expression` with the same signature as execute_cpu
// but with a typed result and without predecessor data
#define ZEROARY_EXECUTE_IMPL                                                   \
	switch (node->operation.data_type) {                                       \
	case F_INT32: {                                                            \
		zeroary_expression(node, (int *__restrict__)result, from, size);       \
	} break;                                                                   \
	case F_FLOAT32: {                                                          \
		zeroary_expression(node, (float *__restrict__)result, from, size);     \
	} break;                                                                   \
	case F_INT64: {                                                            \
		zeroary_expression(node, (long *__restrict__)result, from, size);      \
	} break;                                                                   \
	case F_FLOAT64: {                                                          \
		zeroary_expression(node, (double *__restrict__)result, from, size);    \
	} break;                                                                   \
	}
#define DISPATCH_UNARY_OPERATION(T)                                            \
	switch (node->operation.data_type) {                                       \
	case F_INT32: {                                                            \
		unary_expression((T *__restrict__)result,                              \
						 (int *__restrict__)predecessor_data[0].data, from,    \
						 size, node);                                          \
	} break;                                                                   \
	case F_FLOAT32: {                                                          \
		unary_expression((T *__restrict__)result,                              \
						 (float *__restrict__)predecessor_data[0].data, from,  \
						 size, node);                                          \
	} break;                                                                   \
	case F_INT64: {                                                            \
		unary_expression((T *__restrict__)result,                              \
						 (long *__restrict__)predecessor_data[0].data, from,   \
						 size, node);                                          \
	} break;                                                                   \
	case F_FLOAT64: {                                                          \
		unary_expression((T *__restrict__)result,                              \
						 (double *__restrict__)predecessor_data[0].data, from, \
						 size, node);                                          \
	} break;                                                                   \
	}
/* calls a function `binary_expression` with the signature
 * template <typename T, typename A>
 * void unaryExpression(T *__restrict__ result,
 * 						const A *__restrict__ data1,
 * 						size_t from, size_t size,
 * 						const FGraphNode *curr); */
#define UNARY_EXECUTE_IMPL                                                     \
	switch (node->operation.data_type) {                                       \
	case F_INT32: {                                                            \
		DISPATCH_UNARY_OPERATION(int)                                          \
	} break;                                                                   \
	case F_FLOAT32: {                                                          \
		DISPATCH_UNARY_OPERATION(float)                                        \
	} break;                                                                   \
	case F_INT64: {                                                            \
		DISPATCH_UNARY_OPERATION(long)                                         \
	} break;                                                                   \
	case F_FLOAT64: {                                                          \
		DISPATCH_UNARY_OPERATION(double)                                       \
	} break;                                                                   \
	}

struct OperationImplementation {
		static std::vector<OperationImplementation *> implementations;
		virtual void execute_cpu(const FGraphNode *node,
								 std::vector<CPUResultData> predecessor_data,
								 void *__restrict__ result, size_t from,
								 size_t size) = 0;
		virtual void generate_gpu_lazy() = 0;
		virtual void generate_gpu_eager() = 0;

	protected:
};

#endif
