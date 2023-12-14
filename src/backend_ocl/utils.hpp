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
#ifndef OCL_UTILS_HPP
#define OCL_UTILS_HPP
#include "../../flint.h"
#include "src/errors.hpp"
#include <CL/cl.h>
#include <cmath>
#include <iostream>
#include <list>
#include <mutex>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <tuple>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>
template <typename T>
static cl_mem pushArray(int size, T *data, cl_kernel kernel, cl_context context,
						int &par_index) {
	cl_int err_code;
	cl_mem acc_mem =
		clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					   size * sizeof(T), data, &err_code);
	if (!acc_mem) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
							  std::to_string(err_code));
		return nullptr;
	}
	if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&acc_mem) !=
		CL_SUCCESS) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
							  std::to_string(err_code));
		return nullptr;
	}
	return acc_mem;
}
static cl_mem calcAndPushAccSize(int dim, size_t *shape, cl_kernel kernel,
								 cl_context context, int &par_index) {
	std::vector<size_t> acc_sizes(dim);
	acc_sizes[dim - 1] = 1;
	for (long d = dim - 2; d >= 0; d--) {
		acc_sizes[d] = acc_sizes[d + 1] * shape[d + 1];
	}
	return pushArray(acc_sizes.size(), acc_sizes.data(), kernel, context,
					 par_index);
}
inline void push_per_parameter_dimension(FOperation op, cl_kernel kernel,
										 int &par_index) {
	if (clSetKernelArg(kernel, par_index++, sizeof(int),
					   (void *)&op.dimensions) != CL_SUCCESS) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Could not load Argument to kernel!");
		return;
	}
}
// values for a single operation (not related directly to parameters)
// parameters per operand
inline void pushParameterVals(FGraphNode *node, FGraphNode *pred,
							  cl_kernel kernel, cl_context context,
							  int &par_index, std::list<cl_mem> &to_free) {
	cl_int err_code;
	FOperation op = pred->operation;
	switch (node->operation.op_type) {
	case FPOOLING_SUM:
	case FPOOLING_MAX:
	case FSET_INDEX:
	case FINDEX:
	case FMATMUL:
	case FGRADIENT_POOLING_MAX:
	case FGRADIENT_CONVOLVE1:
	case FGRADIENT_CONVOLVE2:
	case FCONVOLVE: {
		if (clSetKernelArg(kernel, par_index++, sizeof(int),
						   (void *)&op.dimensions) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
	} break;
	case FREDUCE_MIN:
	case FREDUCE_MAX:
	case FREDUCE_SUM:
	case FREDUCE_MUL: {
		int dim = ((int *)node->operation.additional_data)[0];
		const FOperation pred = node->predecessors[0]->operation;
		long it_dim =
			1; // iteration size <=> product of all dimensions along dim
		for (size_t d = dim + 1; d < pred.dimensions; d++)
			it_dim *= pred.shape[d];
		const long shape_dim = pred.shape[dim];
		if (clSetKernelArg(kernel, par_index++, sizeof(int),
						   (void *)&op.dimensions) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		if (clSetKernelArg(kernel, par_index++, sizeof(long),
						   (void *)&it_dim) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		if (clSetKernelArg(kernel, par_index++, sizeof(long),
						   (void *)&shape_dim) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
	} break;
	case FTRANSPOSE: {
		if (clSetKernelArg(kernel, par_index++, sizeof(int),
						   (void *)&op.dimensions) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		std::vector<long> acc_sizes_s(op.dimensions);
		acc_sizes_s[op.dimensions - 1] = 1;
		for (int dim = op.dimensions - 2; dim >= 0; dim--) {
			acc_sizes_s[dim] = acc_sizes_s[dim + 1] * op.shape[dim + 1];
		}
		const int *transpositions = (int *)node->operation.additional_data;
		std::vector<long> acc_sizes_st(op.dimensions);
		for (int i = 0; i < op.dimensions; i++) {
			acc_sizes_st[i] = acc_sizes_s[transpositions[i]];
		}
		to_free.push_back(calcAndPushAccSize(node->operation.dimensions,
											 node->operation.shape, kernel,
											 context, par_index));
		cl_mem ass_mem = clCreateBuffer(
			context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			op.dimensions * sizeof(long), acc_sizes_st.data(), &err_code);
		if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
						   (void *)&ass_mem) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		if (!ass_mem) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		to_free.push_back(ass_mem);
	} break;
	case FSLICE: {
		if (clSetKernelArg(kernel, par_index++, sizeof(int),
						   (void *)&op.dimensions) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		FSlice *slice = (FSlice *)node->operation.additional_data;
		// flattened shape data
		std::vector<size_t> acc_sizes_pred(node->operation.dimensions);
		for (long d = node->operation.dimensions - 1; d >= 0; d--) {
			if (d == node->operation.dimensions - 1)
				acc_sizes_pred[d] = 1;
			else
				acc_sizes_pred[d] = acc_sizes_pred[d + 1] * op.shape[d + 1];
		}
		// allocate steps
		cl_mem steps = clCreateBuffer(
			context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			op.dimensions * sizeof(long), slice->step, &err_code);
		if (!steps)
			flogging(F_ERROR,
					 "Could not load Argument to kernel! Error Code: " +
						 std::to_string(err_code));
		// calculate start and step size in flattened array
		long start = 0;
		for (unsigned int d = 0; d < node->operation.dimensions; d++) {
			start += slice->start[d] * acc_sizes_pred[d];
		}
		to_free.push_back(calcAndPushAccSize(node->operation.dimensions,
											 node->operation.shape, kernel,
											 context, par_index));
		to_free.push_back(calcAndPushAccSize(op.dimensions, op.shape, kernel,
											 context, par_index));
		if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
						   (void *)&steps) != CL_SUCCESS ||
			clSetKernelArg(kernel, par_index++, sizeof(long), (void *)&start) !=
				CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		to_free.push_back(steps);
	} break;
	case FSLIDING_WINDOW: {
		const FOperation pred = node->predecessors[0]->operation;
		const FSlidingWindow *slidewin =
			(FSlidingWindow *)node->operation.additional_data;
		size_t acc_size = node->operation.shape[1];
		std::vector<size_t> acc_sizes_win(pred.dimensions);
		std::vector<size_t> acc_sizes_rest(pred.dimensions);
		acc_sizes_win[acc_sizes_win.size() - 1] = 1;
		acc_sizes_rest[acc_sizes_win.size() - 1] = 1;
		for (int i = acc_sizes_win.size() - 2; i >= 0; i--) {
			acc_size *= node->operation.shape[i + 2];
			acc_sizes_rest[i] = acc_sizes_rest[i + 1] * slidewin->size[i + 1];
			// no of windows in that dimension
			size_t window_size = pred.shape[i + 1] - slidewin->size[i + 1] + 1;
			window_size = window_size % slidewin->step[i + 1] == 0
							  ? window_size / slidewin->step[i + 1]
							  : window_size / slidewin->step[i + 1] + 1;
			acc_sizes_win[i] = acc_sizes_win[i + 1] * window_size;
		}
		if (clSetKernelArg(kernel, par_index++, sizeof(int),
						   (void *)&op.dimensions) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		cl_mem acc_win_mem = clCreateBuffer(
			context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			pred.dimensions * sizeof(long), acc_sizes_win.data(), &err_code);
		if (!acc_win_mem)
			flogging(F_ERROR,
					 "Could not load Argument to kernel! Error Code: " +
						 std::to_string(err_code));
		cl_mem acc_rest_mem = clCreateBuffer(
			context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			pred.dimensions * sizeof(long), acc_sizes_rest.data(), &err_code);
		if (!acc_rest_mem)
			flogging(F_ERROR,
					 "Could not load Argument to kernel! Error Code: " +
						 std::to_string(err_code));
		cl_mem steps = clCreateBuffer(
			context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			pred.dimensions * sizeof(unsigned int), slidewin->step, &err_code);
		if (!steps)
			flogging(F_ERROR,
					 "Could not load Argument to kernel! Error Code: " +
						 std::to_string(err_code));
		to_free.push_back(calcAndPushAccSize(pred.dimensions, pred.shape,
											 kernel, context, par_index));
		if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), &acc_win_mem) !=
			CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
						   &acc_rest_mem) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		if (clSetKernelArg(kernel, par_index++, sizeof(long), &acc_size) !=
			CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), &steps) !=
			CL_SUCCESS)
			flogging(F_ERROR, "Could not load Arguments to kernel!");
		to_free.push_back(acc_win_mem);
		to_free.push_back(acc_rest_mem);
		to_free.push_back(steps);
	} break;
	case FUNSLIDE_WINDOW: {
		const FOperation pred = node->predecessors[0]->operation;
		unsigned int *steps = (unsigned int *)node->operation.additional_data;
		// dimensions 0
		if (clSetKernelArg(kernel, par_index++, sizeof(int),
						   (void *)&op.dimensions) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		// shapeR
		cl_mem shapeR =
			clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						   node->operation.dimensions * sizeof(long),
						   node->operation.shape, &err_code);
		if (!shapeR)
			flogging(F_ERROR,
					 "Could not load Argument to kernel! Error Code: " +
						 std::to_string(err_code));
		if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), &shapeR) !=
			CL_SUCCESS)
			flogging(F_ERROR, "Could not load Argument to kernel! 0");
		to_free.push_back(shapeR);
		// acc_sizes
		to_free.push_back(calcAndPushAccSize(node->operation.dimensions,
											 node->operation.shape, kernel,
											 context, par_index));
		// shapeR
		cl_mem shape0 = clCreateBuffer(
			context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			pred.dimensions * sizeof(long), pred.shape, &err_code);
		if (!shape0)
			flogging(F_ERROR,
					 "Could not load Argument to kernel! Error Code: " +
						 std::to_string(err_code));
		if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), &shape0) !=
			CL_SUCCESS)
			flogging(F_ERROR, "Could not load Argument to kernel! 1");
		to_free.push_back(shape0);
		// acc_sizes_pred
		to_free.push_back(calcAndPushAccSize(pred.dimensions, pred.shape,
											 kernel, context, par_index));
		size_t no_windows[pred.dimensions - 1];
		for (int i = 0; i < pred.dimensions - 1; i++) {
			size_t window_size =
				node->operation.shape[i] - pred.shape[i + 1] + 1;
			no_windows[i] = window_size % steps[i] == 0
								? window_size / steps[i]
								: window_size / steps[i] + 1;
		}
		// acc_no_windows
		to_free.push_back(calcAndPushAccSize(pred.dimensions - 1, no_windows,
											 kernel, context, par_index));
		// no_windows
		cl_mem windows = clCreateBuffer(
			context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			(pred.dimensions - 1) * sizeof(long), no_windows, &err_code);
		if (!windows)
			flogging(F_ERROR,
					 "Could not load Argument to kernel! Error Code: " +
						 std::to_string(err_code));
		if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), &windows) !=
			CL_SUCCESS)
			flogging(F_ERROR, "Could not load Argument to kernel! 2");
		to_free.push_back(windows);
		// steps
		cl_mem steps_mem = clCreateBuffer(
			context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			(pred.dimensions - 1) * sizeof(unsigned int), steps, &err_code);
		if (!steps_mem)
			flogging(F_ERROR,
					 "Could not load Argument to kernel! Error Code: " +
						 std::to_string(err_code));
		if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), &steps_mem) !=
			CL_SUCCESS)
			flogging(F_ERROR, "Could not load Arguments to kernel! 3");
		to_free.push_back(steps_mem);
	} break;
	case FREPEAT: {
		if (clSetKernelArg(kernel, par_index++, sizeof(int),
						   (void *)&op.dimensions) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		to_free.push_back(calcAndPushAccSize(node->operation.dimensions,
											 node->operation.shape, kernel,
											 context, par_index));
		to_free.push_back(calcAndPushAccSize(op.dimensions, op.shape, kernel,
											 context, par_index));
		cl_mem predshape_mem =
			clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						   op.dimensions * sizeof(long), op.shape, &err_code);
		if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
						   (void *)&predshape_mem) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		to_free.push_back(predshape_mem);
	} break;
	case FEXTEND: {
		if (clSetKernelArg(kernel, par_index++, sizeof(int),
						   (void *)&op.dimensions) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		to_free.push_back(calcAndPushAccSize(node->operation.dimensions,
											 node->operation.shape, kernel,
											 context, par_index));
		to_free.push_back(calcAndPushAccSize(op.dimensions, op.shape, kernel,
											 context, par_index));
		const FExtend *extend = (FExtend *)node->operation.additional_data;
		cl_mem steps_mem = clCreateBuffer(
			context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			op.dimensions * sizeof(long), extend->step, &err_code);
		if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
						   (void *)&steps_mem) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		cl_mem start_mem = clCreateBuffer(
			context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			op.dimensions * sizeof(long), extend->start, &err_code);
		if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
						   (void *)&start_mem) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		cl_mem predshape_mem =
			clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						   op.dimensions * sizeof(long), op.shape, &err_code);
		if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
						   (void *)&predshape_mem) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return;
		}
		to_free.push_back(steps_mem);
		to_free.push_back(start_mem);
		to_free.push_back(predshape_mem);
	} break;
	default:
		break;
	}
}
/** Returns a map mapping to each node in the subgraph of root the number of
 * incoming edges, i.e. the number of nodes that have that node as parent */
inline std::unordered_map<FGraphNode *, int>
calculateNumEdges(FGraphNode *root) {
	std::list<FGraphNode *> todo;
	std::unordered_map<FGraphNode *, int> num_edges;
	todo.push_front(root);
	num_edges.insert({root, 0});
	while (!todo.empty()) {
		FGraphNode *c = todo.front();
		todo.pop_front();
		for (int i = 0; i < c->num_predecessor; i++) {
			if (num_edges.find(c->predecessors[i]) == num_edges.end()) {
				num_edges.insert({c->predecessors[i], 1});
				todo.push_front(c->predecessors[i]);
			} else
				num_edges[c->predecessors[i]]++;
		}
	}
	return num_edges;
}
/** Calculates a topological sort of the operational graph
 * with Kahns algorithm */
inline std::list<FGraphNode *> topologicalSort(FGraphNode *root) {
	std::list<FGraphNode *> result;
	std::unordered_map<FGraphNode *, int> num_edges = calculateNumEdges(root);
	std::list<FGraphNode *> no_incoming;
	no_incoming.push_back(root);
	while (!no_incoming.empty()) {
		FGraphNode *n = no_incoming.front();
		result.push_back(n);
		no_incoming.pop_front();
		for (int i = 0; i < n->num_predecessor; i++) {
			if (--num_edges[n->predecessors[i]] == 0)
				no_incoming.push_back(n->predecessors[i]);
		}
	}
	return result;
}
#endif
