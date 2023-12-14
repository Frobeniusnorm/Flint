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
#ifndef FLINT_IMPLEMENTATIONS_HPP
#define FLINT_IMPLEMENTATIONS_HPP

#include "../../flint.h"
#include "../backend_cpu/cpu_common.hpp"
#include "../backend_ocl/twine.hpp"
#include "../utils.hpp"
#include <unordered_map>
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
	switch (node->predecessors[0]->operation.data_type) {                      \
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
/* calls a function `unary_expression` with the signature
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
/* calls a function `unary_expression` with the signature
 * template <typename T>
 * void unaryExpression(T *__restrict__ result,
 * 						const T *__restrict__ data1,
 * 						size_t from, size_t size,
 * 						const FGraphNode *curr); */
#define UNARY_EXECUTE_MONOTON_IMPL                                             \
	switch (node->operation.data_type) {                                       \
	case F_INT32: {                                                            \
		unary_expression((int *__restrict__)result,                            \
						 (int *__restrict__)predecessor_data[0].data, from,    \
						 size, node);                                          \
	} break;                                                                   \
	case F_FLOAT32: {                                                          \
		unary_expression((float *__restrict__)result,                          \
						 (float *__restrict__)predecessor_data[0].data, from,  \
						 size, node);                                          \
	} break;                                                                   \
	case F_INT64: {                                                            \
		unary_expression((long *__restrict__)result,                           \
						 (long *__restrict__)predecessor_data[0].data, from,   \
						 size, node);                                          \
	} break;                                                                   \
	case F_FLOAT64: {                                                          \
		unary_expression((double *__restrict__)result,                         \
						 (double *__restrict__)predecessor_data[0].data, from, \
						 size, node);                                          \
	} break;                                                                   \
	}
/* calls a function `execute_cpu_typed` with the same signature as
 * `execute_cpu`, but with a typed return type */
#define EXECUTE_TYPED_IMPL                                                     \
	switch (node->operation.data_type) {                                       \
	case F_INT32: {                                                            \
		execute_cpu_typed(node, predecessor_data, (int *__restrict__)result,   \
						  from, size);                                         \
	} break;                                                                   \
	case F_FLOAT32: {                                                          \
		execute_cpu_typed(node, predecessor_data, (float *__restrict__)result, \
						  from, size);                                         \
	} break;                                                                   \
	case F_INT64: {                                                            \
		execute_cpu_typed(node, predecessor_data, (long *__restrict__)result,  \
						  from, size);                                         \
	} break;                                                                   \
	case F_FLOAT64: {                                                          \
		execute_cpu_typed(node, predecessor_data,                              \
						  (double *__restrict__)result, from, size);           \
	} break;                                                                   \
	}
/* calls a function `binary_expression` with the signature
 * template <typename T>
 * void binaryExpression(T *__restrict__ result,
 * 						const T *__restrict__ data1,
 * 						const T *__restrict__ data2, size_t from,
 * 						size_t size, size_t index_man_1,
 * 						size_t inv_man_1, size_t index_man_2,
 * 						size_t inv_man_2, const FGraphNode *curr); */
#define BINARY_EXECUTE_MONOTON_IMPL                                            \
	const CPUResultData p1 = predecessor_data[0], p2 = predecessor_data[1];    \
	size_t im1 = p1.num_entries, im2 = p2.num_entries;                         \
	size_t iv1 = 1, iv2 = 1;                                                   \
	calculateDivisorForInverseBroadcasting(node->predecessors[0], iv1,         \
										   node->predecessors[1], iv2);        \
	switch (node->operation.data_type) {                                       \
	case F_INT32: {                                                            \
		binary_expression((int *)result, (int *)p1.data, (int *)p2.data, from, \
						  size, im1, iv1, im2, iv2, node);                     \
	} break;                                                                   \
	case F_FLOAT32: {                                                          \
		binary_expression((float *)result, (float *)p1.data, (float *)p2.data, \
						  from, size, im1, iv1, im2, iv2, node);               \
	} break;                                                                   \
	case F_INT64: {                                                            \
		binary_expression((long *)result, (long *)p1.data, (long *)p2.data,    \
						  from, size, im1, iv1, im2, iv2, node);               \
	} break;                                                                   \
	case F_FLOAT64: {                                                          \
		binary_expression((double *)result, (double *)p1.data,                 \
						  (double *)p2.data, from, size, im1, iv1, im2, iv2,   \
						  node);                                               \
	} break;                                                                   \
	}
struct OCLLazyCodegenState {
		/** Working queue of nodes for which still code has to be generated */
		std::list<std::tuple<FGraphNode *, std::string>> todo;
		/** Maps storage nodes to their kernel parameter names (for fast lookup)
		 */
		std::unordered_map<FGraphNode *, std::string> assigned_params;
		/** To register new parameter nodes to their name */
		std::list<std::pair<FGraphNode *, std::string>> *parameters;
		/** indexing logic (we save the old index in old_index$i to restore it)
		 */
		unsigned int num_indices = 0;
		/** Number of nodes that are already assigned to variables (for naming)
		 */
		unsigned int variable_index = 0;
		/** a code segment that is inserted before the predecessors (cleared
		 * after each node) */
		std::string index_defs;
		/** Actual code as twine for fast prepend and append operations */
		Twine code;
		/**
		 * Checks if the nodes has already been included as a parameter for the
		 * kernel. If it has, returnes the binded variable, else it creates a
		 * new parameter for it and returns its name.
		 */
		std::string findOrInsertParameter(FGraphNode *gnp1) {
			std::string par1;
			if (assigned_params.find(gnp1) != assigned_params.end()) {
				par1 = assigned_params[gnp1];
			} else {
				par1 = "P" + std::to_string(assigned_params.size());
				assigned_params.insert({gnp1, par1});
				parameters->push_back({gnp1, par1});
			}
			return par1;
		}
};
struct OperationImplementation {
		/** Disables automatic code generation for the parents */
		static const int OCL_LAZY_DONT_PUSH_PREDS = 1;
		/** Enables automatic index insertion for inverse broadcasting if the
		 * node request it */
		static const int OCL_LAZY_INVERSE_BROADCASTING = 2;
		static std::vector<OperationImplementation *> implementations;
		/** Executes the given node in the range of `from` to `from + size` and
		 * stores its result data in `result`. The results of the parameters are
		 * in `predecessor_data`. See Dispatch and Execute macros to dispatch to
		 * typed helper functions. */
		virtual void execute_cpu(const FGraphNode *node,
								 std::vector<CPUResultData> predecessor_data,
								 void *__restrict__ result, size_t from,
								 size_t size) = 0;
		/** Prepends the generated gpu lazy code for this operation to `code`.
		 * The return value should be generated by using the `OCL_LAZY_`...
		 * flags. Multiple flags are combined with binary or. `name` is the name
		 * of the variable that is to be generated for `node`. `compiler_state`
		 * is a set of variables alive during the complete kernel generation,
		 * here the definition of the `code` string is as well*/
		virtual int generate_ocl_lazy(const FGraphNode *node, std::string name,
									  OCLLazyCodegenState &compiler_state) = 0;
		/**
		 * Generates the content of a eager kernel.
		 * `res_type` is the result type of the kernel and `parameter_types`
		 * contains the parameter types in order.
		 */
		virtual std::string
		generate_ocl_eager(FType res_type,
						   std::vector<FType> parameter_types) = 0;
		/**
		 * Generates additional parameters to the kernel additional to the
		 * result array
		 */
		virtual std::string
		generate_ocl_parameters_eager(FType res_type,
									  std::vector<FType> parameter_types) {
			Twine code;
			for (int i = 0; i < parameter_types.size(); i++)
				code += ", const __global " + typeString(parameter_types[i]) +
						"* P" + std::to_string(i) + ", long num_entries" +
						std::to_string(i);
			return code;
		}
		/**
		 * Pushes additional values to the eager opencl program that don't
		 * depend on the parameters. The `par_index` has to be incremented for
		 * every pushed parameter. Every memory object in `to_free` will be
		 * freed after program execution.
		 */
		virtual void
		push_additional_kernel_parameters(FGraphNode *node, cl_kernel kernel,
										  cl_context context, int &par_index,
										  std::list<cl_mem> &to_free) {}
		/**
		 * Pushes per parameter values (the function is called once per
		 * parameter). See `push_additional_kernel_parameters`.
		 */
		virtual void push_parameter_kernel_parameters(
			FGraphNode *node, FGraphNode *pred, cl_kernel kernel,
			cl_context context, int &par_index, std::list<cl_mem> &to_free) {}
		/**
		 * Calculates the operation score for a node, i.e. assigns a score
		 * to each node depending on its parallelizability, very high scores are
		 * calculated on gpu, middle high scores parallel on cpus, lower scores
		 * sequentially. The scores is multiplied with the number of elements of
		 * the node (assuming a node has 5000 elements and `operation_score`
		 * returns 2, the final score is 10000)
		 */
		virtual int operation_score(FGraphNode *node) {
      return 2;
    }
};
#endif
