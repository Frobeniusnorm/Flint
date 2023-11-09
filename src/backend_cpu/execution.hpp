/* Copyright 2022 David Schwarzbeck

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

  This file includes the implementation of the CPU backend.
*/
#ifndef CPU_BACKEND_EXECUTION
#define CPU_BACKEND_EXECUTION
#include "../../flint.h"
#include "../utils.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <ostream>
#include <vector>
#define MIN_VAL(x, y) x < y ? x : y
#define MAX_VAL(x, y) x < y ? y : x
// TODO capability to lazily compute some operations
// use generators for that and overload CPUResultDatas capabilities.
// e.g. transpose saves a overloaded Generator struct whith all the relevant
// information to compute a index.
struct CPUResultData {
		void *data;
		FType type;
		size_t num_entries;
		std::vector<size_t> shape;
};
template <typename T, typename A>
static void unaryExpression(T *__restrict__ result, const A *__restrict__ data,
							FOperationType op, size_t from, size_t size,
							int index_man, const FGraphNode *curr) {
	switch (op) {
	case FSIGN: {
		for (size_t i = from; i < from + size; i++)
			result[i] = data[i] < 0 ? -1 : 1;
	} break;
	case FEVEN: { // it aint pretty but it does its job
		for (size_t i = from; i < from + size; i++) {
			if constexpr (std::is_same<A, int>() || std::is_same<A, long>())
				result[i] = data[i] % 2 == 0 ? 1 : 0;
			else
				result[i] = 0;
		}
	} break;
	case FCONVERSION: {
		for (size_t i = from; i < from + size; i++)
			result[i] = (T)data[i];

	} break;
	case FRESHAPE:
	case FLATTEN: {
		for (size_t i = from; i < from + size; i++)
			result[i] = data[i];
	} break;
	case FPOOLING_MAX:
	case FPOOLING_SUM: {
		const FOperation op = curr->operation;
		const FGraphNode *gnp1 = curr->predecessors[0];
		const FOperation pred = gnp1->operation;
		const FSlidingWindow *window = (FSlidingWindow *)op.additional_data;
		// calculate accumulated sizes for result, kernel and source (pred)
		const std::vector<size_t> acc_sizes = calcAccSizes(op);
		const std::vector<size_t> acc_sizes_pred = calcAccSizes(pred);
		size_t kernel_num_elems = window->size[op.dimensions - 1];
		std::vector<size_t> acc_sizes_kernel =
			std::vector<size_t>(op.dimensions);
		acc_sizes_kernel[op.dimensions - 1] = 1;
		for (int d = op.dimensions - 2; d >= 0; d--) {
			acc_sizes_kernel[d] = acc_sizes_kernel[d + 1] * window->size[d + 1];
			kernel_num_elems *= window->size[d];
		}

		for (size_t i = from; i < from + size; i++) {
			// base index for source
			size_t j = 0;
			for (unsigned int d = 0; d < op.dimensions; d++) {
				// get dimension index
				const size_t di =
					(d == 0 ? i : i % acc_sizes[d - 1]) / acc_sizes[d];
				// reproject
				j += di * window->step[d] * acc_sizes_pred[d];
			}
			T res = 0;
			for (size_t k = 0; k < kernel_num_elems; k++) {
				bool set_zero = false;
				size_t o = 0; // source offset
				for (unsigned int d = 0; d < op.dimensions; d++) {
					const size_t dk =
						(d == 0 ? k : k % acc_sizes_kernel[d - 1]) /
						acc_sizes_kernel[d];
					o += dk * acc_sizes_pred[d];
				}
				// full reduction in last dimension so iterate over it too
				for (size_t ld = 0; ld < pred.shape[pred.dimensions - 1];
					 ld++) {
					switch (op.op_type) {
					case FPOOLING_SUM:
						res += data[j + o + ld];
						break;
					case FPOOLING_MAX:
						res = MAX_VAL(data[j + o + ld], res);
					default:
						break;
					}
				}
			}
			result[i] = res;
		}
	} break;
	default:
		break;
	}
}
template <typename T, typename A, typename B>
static void
binaryExpression(T *__restrict__ result, const A *__restrict__ data1,
				 const B *__restrict__ data2, FOperationType op, size_t from,
				 size_t size, size_t index_man_1, size_t inv_man_1,
				 size_t index_man_2, size_t inv_man_2, const FGraphNode *curr) {
	switch (op) {
	case FADD:
		for (size_t i = from; i < from + size; i++) {
			result[i] = data1[(i / inv_man_1) % index_man_1] +
						data2[(i / inv_man_2) % index_man_2];
		}
		break;
	case FSUB:
		for (size_t i = from; i < from + size; i++)
			result[i] = data1[(i / inv_man_1) % index_man_1] -
						data2[(i / inv_man_2) % index_man_2];
		break;
	case FMUL:
		for (size_t i = from; i < from + size; i++)
			result[i] = data1[(i / inv_man_1) % index_man_1] *
						data2[(i / inv_man_2) % index_man_2];
		break;
	case FDIV:
		for (size_t i = from; i < from + size; i++)
			result[i] = data1[(i / inv_man_1) % index_man_1] /
						data2[(i / inv_man_2) % index_man_2];
		break;
	case FPOW:
		for (size_t i = from; i < from + size; i++)
			result[i] = pow(data1[(i / inv_man_1) % index_man_1],
							data2[(i / inv_man_2) % index_man_2]);
		break;
	case FMATMUL: {
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
	} break;
	case FCONCAT: {
		FGraphNode *a = curr->predecessors[0];
		FGraphNode *b = curr->predecessors[1];
		unsigned int ax = ((unsigned int *)curr->operation.additional_data)[0];
		size_t acc_size_last = 1;
		for (int i = curr->operation.dimensions - 2; i >= (int)ax; i--) {
			acc_size_last *= curr->operation.shape[i + 1];
		}
		for (size_t index = from; index < from + size; index++) {
			size_t sx = index / acc_size_last;
			size_t sc = ax > 0 ? sx % curr->operation.shape[ax] : sx;
			// reproject to one of the tensors
			if (sc < a->operation.shape[ax]) {
				size_t ai = (sx / curr->operation.shape[ax]) * acc_size_last *
								a->operation.shape[ax] +
							sc * acc_size_last + index % acc_size_last;
				result[index] = data1[ai];
			} else {
				size_t bi = (sx / curr->operation.shape[ax]) * acc_size_last *
								b->operation.shape[ax] +
							(sc - a->operation.shape[ax]) * acc_size_last +
							index % acc_size_last;
				result[index] = data2[bi];
			}
		}
	} break;
	case FGRADIENT_CONVOLVE1: {
		const FOperation op = curr->operation;
		const FGraphNode *gnp1 = curr->predecessors[0],
						 *gnp2 = curr->predecessors[1];
		const FOperation kernel = gnp1->operation, a = gnp2->operation;
		const unsigned int *steps = (unsigned int *)op.additional_data;
		// calculate accumulated sizes for result (pred), kernel and a
		// (adjacent)
		std::vector<size_t> acc_sizes(op.dimensions - 1);
		std::vector<size_t> acc_sizes_pred = calcAccSizes(op);
		std::vector<size_t> acc_sizes_kernel = calcAccSizes(kernel);
		acc_sizes[op.dimensions - 2] = 1;
		size_t kernel_num_elems = kernel.shape[op.dimensions - 1];
		size_t a_num_elems = 1;
		for (long d = a.dimensions - 1; d >= 0; d--)
			a_num_elems *= a.shape[d];
		for (long d = op.dimensions - 2; d >= 0; d--)
			kernel_num_elems *= kernel.shape[d];
		for (long d = op.dimensions - 3; d >= 0; d--)
			acc_sizes[d] = acc_sizes[d + 1] * a.shape[d + 1];
		for (size_t i = from; i < from + size; i++) {
			T res = 0;
			long k = 0;
			bool in_steps = true;
			// reproject first time kernel hits i
			for (int d = op.dimensions - 1; d >= 0; d--) {
				size_t di = (d == 0 ? i : i % acc_sizes_pred[d - 1]) /
							acc_sizes_pred[d];
				size_t dk = d == op.dimensions - 1 ? di : di % steps[d];
				if (dk >= kernel.shape[d]) {
					in_steps = false;
					break;
				}
				k += dk * acc_sizes_kernel[d];
			}
			if (in_steps)
				while (k < kernel_num_elems) {
					size_t i_conv = 0;
					for (int d = 0; d < op.dimensions - 2; d++) {
						const size_t dk =
							(d == 0 ? k : k % acc_sizes_kernel[d - 1]) /
							acc_sizes_kernel[d];
						const size_t di =
							(d == 0 ? i : i % acc_sizes_pred[d - 1]) /
							acc_sizes_pred[d];
						const size_t j =
							(di - dk) / steps[d]; // top left corner
						i_conv += j * acc_sizes[d];
					}
					if (i_conv < a_num_elems)
						res += data1[k] * data2[i_conv];
					long step = 0;
					// reproject index to calculate step from steps
					for (int d = op.dimensions - 2; d >= 0; d--) {
						const int stepd = steps[d];
						const size_t dk =
							(d == 0 ? k : k % acc_sizes_kernel[d - 1]) /
							acc_sizes_kernel[d];
						const size_t di =
							(d == 0 ? i : i % acc_sizes_pred[d - 1]) /
							acc_sizes_pred[d];
						if (dk + stepd < kernel.shape[d] && di >= dk + stepd) {
							step += stepd * acc_sizes_kernel[d];
							break;
						} else {
							step -= (dk - (di % stepd)) *
									acc_sizes_kernel[d]; // set to kernel start
														 // in this dimension
						}
					}
					if (step <= 0)
						break; // total overflow
					k += step;
				}
			result[i] = res;
		}
	} break;
	case FCONVOLVE: {
		const FOperation op = curr->operation;
		const FGraphNode *gnp1 = curr->predecessors[0],
						 *gnp2 = curr->predecessors[1];
		const FOperation pred = gnp1->operation, kernel = gnp2->operation;
		const unsigned int *steps = (unsigned int *)op.additional_data;
		const bool multiple_filter =
			gnp2->operation.dimensions != gnp1->operation.dimensions;
		// calculate accumulated sizes for result, kernel and source (pred)
		std::vector<size_t> acc_sizes = calcAccSizes(op);
		std::vector<size_t> acc_sizes_pred = calcAccSizes(pred);
		std::vector<size_t> acc_sizes_kernel = calcAccSizes(kernel);
		size_t kernel_num_elems = kernel.shape[acc_sizes.size()];
		size_t pred_num_elems =
			multiple_filter ? 1 : pred.shape[acc_sizes.size()];
		for (long d = acc_sizes.size() - 1; d >= 0; d--) {
			pred_num_elems *= pred.shape[d];
			if (d != 0 || !multiple_filter) // since kernel.shape[0] is the
											// dimension of filters
				kernel_num_elems *= kernel.shape[d];
		}
		for (size_t i = from; i < from + size; i++) {
			size_t j = 0;
			// we can ignore last index of source and kernel for result since we
			// iterate over it (i.e. for the destination it is 0 since it does
			// not have that dimension)
			for (unsigned int d = 0;
				 d < (multiple_filter ? op.dimensions - 1 : op.dimensions);
				 d++) {
				// get dimension index
				size_t di = (d == 0 ? i : i % acc_sizes[d - 1]) / acc_sizes[d];
				// reproject
				j += di * steps[d] * acc_sizes_pred[d];
			}
			// we must offset the kernel by the filter index, which is the last
			// dimension of the result
			size_t kernel_offset = 0;
			if (multiple_filter) {
				// filter index
				size_t fi = (i % acc_sizes[op.dimensions - 2]) /
							acc_sizes[op.dimensions - 1];
				kernel_offset =
					fi * kernel_num_elems; // since the filters are the
			}
			// now that we have the correct base index in source, convolve
			T res = 0;
			for (size_t k = 0; k < kernel_num_elems; k++) {
				bool set_zero = false;
				size_t o = 0; // source offset
				// reproject kernel
				const unsigned int last_dim = multiple_filter
												  ? acc_sizes_kernel.size() - 1
												  : acc_sizes_kernel.size();
				for (unsigned int d = 0; d < last_dim; d++) {
					const unsigned int kn_d = multiple_filter ? d + 1 : d;
					size_t di = 0;
					if (d != last_dim - 1)
						di = (d == 0 ? i : i % acc_sizes[d - 1]) / acc_sizes[d];
					size_t dk =
						(kn_d == 0 ? k : k % acc_sizes_kernel[kn_d - 1]) /
						acc_sizes_kernel[kn_d];
					if (d < pred.dimensions - 1)
						if (((di * steps[d]) + dk) * acc_sizes_pred[d] >=
								pred_num_elems ||
							(d > 0 &&
							 ((di * steps[d]) + dk) * acc_sizes_pred[d] >=
								 acc_sizes_pred[d - 1])) {
							set_zero = true;
							break;
						}
					o += dk * acc_sizes_pred[d];
				}
				if (set_zero)
					continue;
				res += data2[k + kernel_offset] * data1[j + o];
			}
			result[i] = res;
		}
	} break;
	case FGRADIENT_CONVOLVE2: {
		// normal convolution:
		//   shape(op) = [k1, k2, ..., kn, c]
		//   shape(pred) = [p1, p2, ..., pn, c]
		//   shape(prev_adj) = [w1, w2, ..., wn]
		// multifilter convolution:
		//   shape(op) = [filter, k1, k2, ..., kn, c]
		//   shape(pred) = [p1, p2, ..., pn, c]
		//   shape(prev_adj) = [w1, w2, ..., wn, filter]
		const FOperation op = curr->operation;
		const FGraphNode *gnp1 = curr->predecessors[0],
						 *gnp2 = curr->predecessors[1];
		const FOperation pred = gnp1->operation, prev_adj = gnp2->operation;
		const std::vector<size_t> acc_sizes_pred = calcAccSizes(pred);
		const std::vector<size_t> acc_sizes_kernel = calcAccSizes(op);
		const bool multifilter = op.dimensions > pred.dimensions;
		// like accumulated sizes for prev_adj but without filter in multifilter
		// context
		std::vector<size_t> acc_sizes_windows(
			multifilter ? prev_adj.dimensions - 1 : prev_adj.dimensions);
		acc_sizes_windows[acc_sizes_windows.size() - 1] = 1;
		for (int i = acc_sizes_windows.size() - 2; i >= 0; i--) {
			acc_sizes_windows[i] =
				acc_sizes_windows[i + 1] * prev_adj.shape[i + 1];
		}
		// total number of windows
		const size_t windows = acc_sizes_windows[0] * prev_adj.shape[0];
		// helper variables
		const size_t num_elems_kernel = multifilter
											? acc_sizes_kernel[0]
											: acc_sizes_kernel[0] * op.shape[0];
		const unsigned int *steps = (unsigned int *)op.additional_data;
		const unsigned int num_filter = multifilter ? op.shape[0] : 1;
		for (size_t i = from; i < from + size; i++) {
			// filter entry of current iteration for multifilter
			size_t f = 0;
			if (multifilter) {
				f = i / num_elems_kernel;
			}
			// project kernel offset to a offset
			size_t a_offset = 0;
			for (int j = multifilter ? 1 : 0; j < op.dimensions; j++) {
				size_t ki = (i / acc_sizes_kernel[j]) % op.shape[j];
				a_offset += ki * acc_sizes_pred[multifilter ? j - 1 : j];
			}
			result[i] = 0;
			// iterate over windows = adjoint elements in first dimensions
			for (size_t w = 0; w < windows; w++) {
				// calculate start value of window for pred
				size_t a = 0;
				for (int j = 0; j < acc_sizes_windows.size(); j++) {
					size_t wj = (w / acc_sizes_windows[j]) % prev_adj.shape[j];
					a += wj * acc_sizes_pred[j] * steps[j];
				}
				result[i] += data1[a + a_offset] * data2[w * num_filter + f];
			}
		}
	} break;
	case FSLIDE: {
		const FOperation op = curr->operation;
		const FGraphNode *gnp1 = curr->predecessors[0],
						 *gnp2 = curr->predecessors[1];
		const FOperation pred = gnp1->operation, kernel = gnp2->operation;
		std::vector<size_t> acc_sizes_pred = calcAccSizes(pred);
		std::vector<size_t> acc_sizes_kernel = calcAccSizes(kernel);
		size_t pred_num_elems = pred.shape[pred.dimensions - 1];
		for (long d = pred.dimensions - 2; d >= 0; d--)
			pred_num_elems *= pred.shape[d];
		const unsigned int *steps = (unsigned int *)op.additional_data;
		for (size_t i = from; i < from + size; i++) {
			size_t a = 0;
			size_t dis[kernel.dimensions];
			// reproject start
			for (int d = kernel.dimensions - 1; d >= 0; d--) {
				dis[d] = (d == 0 ? i : i % acc_sizes_kernel[d - 1]) /
						 acc_sizes_kernel[d];
				a += dis[d] * acc_sizes_pred[d];
			}
			T res = 0;
			// we want to iterate over all elements it would be slid against
			while (a < pred_num_elems) {
				long step = 0;
				res += data1[a] * data2[i];
				// reproject index to calculate step from steps
				for (int d = pred.dimensions - 2; d >= 0; d--) {
					// index in this dimension
					size_t da = (d == 0 ? a : a % acc_sizes_pred[d - 1]) /
								acc_sizes_pred[d];
					if (da + (kernel.shape[d] - dis[d] - 1) + steps[d] <
						pred.shape[d]) {
						step += steps[d] * acc_sizes_pred[d];
						break;
					} else {
						step -= (da - dis[d]) *
								acc_sizes_pred[d]; // set to kernel start in
												   // this dimension
					}
				}
				if (step <= 0)
					break; // total overflow
				a += step;
			}
			result[i] = res;
		}
	} break;
	case FINDEX: {
		const FGraphNode *a = curr->predecessors[0];
		const FGraphNode *b = curr->predecessors[1];
		const FOperation op = curr->operation;
		const unsigned int axis = b->operation.dimensions - 1;
		// get index of result, index tensor, reproject index
		size_t acc_sizes_ax = 1;
		for (int i = axis + 1; i < op.dimensions; i++)
			acc_sizes_ax *= op.shape[i];

		for (size_t i = from; i < from + size; i++) {
			const size_t base = i / (acc_sizes_ax * op.shape[axis]);
			const size_t rest = i % acc_sizes_ax;
			const size_t ind = (size_t)data2[i / acc_sizes_ax];
			result[i] = data1[(base * acc_sizes_ax * a->operation.shape[axis]) +
							  (ind * acc_sizes_ax) + rest];
		}
	} break;
	case FMIN:
		for (size_t i = from; i < from + size; i++)
			result[i] = MIN_VAL(data1[(i / inv_man_1) % index_man_1],
								data2[(i / inv_man_2) % index_man_2]);
		break;
	case FMAX:
		for (size_t i = from; i < from + size; i++)
			result[i] = MAX_VAL(data1[(i / inv_man_1) % index_man_1],
								data2[(i / inv_man_2) % index_man_2]);
		break;
	case FEQUAL:
		for (size_t i = from; i < from + size; i++)
			result[i] = data1[(i / inv_man_1) % index_man_1] ==
								data2[(i / inv_man_2) % index_man_2]
							? 1
							: 0;
		break;
	case FLESS:
		for (size_t i = from; i < from + size; i++)
			result[i] = data1[(i / inv_man_1) % index_man_1] <
								data2[(i / inv_man_2) % index_man_2]
							? 1
							: 0;
		break;
	case FGREATER:
		for (size_t i = from; i < from + size; i++)
			result[i] = data1[(i / inv_man_1) % index_man_1] >
								data2[(i / inv_man_2) % index_man_2]
							? 1
							: 0;
		break;
	default:
		break;
	}
}
// i hate this function more than anything else in this library (yet)
// EDIT: nope i was wrong, computeGradient is worse
template <typename T>
static void executeNode(const FGraphNode *node,
						std::vector<CPUResultData> predecessor_data,
						T *__restrict__ result, size_t from, size_t size) {
	// handle operations that are zero-ary or unary and dont need type
	// parameters
	switch (node->operation.op_type) {
	case FGEN_RANDOM: {
		double seed = ((double *)node->operation.additional_data)[0];
		for (size_t i = from; i < from + size; i++) {
			double v = sin(i + seed) * 43758.5453123;
			result[i] = std::min(v - floor(v), 0.99999);
		}
	} break;
	case FGEN_CONSTANT: {
		T value = ((T *)node->operation.additional_data)[0];
		for (size_t i = from; i < from + size; i++)
			result[i] = value;
	} break;
	case FGEN_ARANGE: {
		unsigned int ax = ((unsigned int *)node->operation.additional_data)[0];
		size_t acc_sizes_ax = 1;
		for (unsigned int i = ax + 1; i < node->operation.dimensions; i++)
			acc_sizes_ax *= node->operation.shape[i];
		for (size_t i = from; i < from + size; i++)
			result[i] = (i / acc_sizes_ax) % node->operation.shape[ax];
	} break;
	case FREPEAT: {
		const FOperation op = node->operation;
		const CPUResultData pred = predecessor_data[0];
		const void *__restrict__ data = pred.data;
		// calculate number of elements per dimension entry for destination and
		// source
		std::vector<size_t> acc_sizes_d = calcAccSizes(op);
		std::vector<size_t> acc_sizes_s =
			calcAccSizes(pred.shape.size(), pred.shape.data());
		for (int i = from; i < from + size; i++) {
			// to get the index in the source array we first calculate the
			// indices and reproject
			size_t index = i;
			size_t src_index = 0;
			for (int dim = 0; dim < op.dimensions; dim++) {
				int curr_idx = index / acc_sizes_d[dim];
				index %= acc_sizes_d[dim];
				src_index += (curr_idx % pred.shape[dim]) * acc_sizes_s[dim];
			}
			result[i] = ((const T *__restrict__)data)[src_index];
		}
	} break;
	case FTRANSPOSE: {
		const FOperation op = node->operation;
		const int *transposition = (int *)op.additional_data;
		CPUResultData pred = predecessor_data[0];
		const void *__restrict__ data = pred.data;
		// calculate number of elements per dimension entry for destination and
		// source
		std::vector<size_t> acc_sizes_d = calcAccSizes(op);
		std::vector<size_t> acc_sizes_s =
			calcAccSizes(pred.shape.size(), pred.shape.data());
		for (int i = from; i < from + size; i++) {
			// to get the index in the source array we first calculate the
			// indices and reproject
			int index = i;
			int src_index = 0;
			for (int dim = 0; dim < op.dimensions; dim++) {
				int curr_idx = index / acc_sizes_d[dim];
				index %= acc_sizes_d[dim];
				src_index += curr_idx * acc_sizes_s[transposition[dim]];
			}
			result[i] = ((const T *__restrict__)data)[src_index];
		}
	} break;
	case FSLIDING_WINDOW: {
		CPUResultData pred = predecessor_data[0];
		const FSlidingWindow *slidewin =
			(FSlidingWindow *)node->operation.additional_data;
		const void *__restrict__ data = pred.data;
		size_t acc_size = node->operation.shape[1];
		std::vector<size_t> acc_sizes_pred =
			calcAccSizes(pred.shape.size(), pred.shape.data());
		std::vector<size_t> acc_sizes_win(pred.shape.size());
		std::vector<size_t> acc_sizes_rest(pred.shape.size());
		acc_sizes_win[acc_sizes_win.size() - 1] = 1;
		acc_sizes_rest[acc_sizes_win.size() - 1] = 1;
		for (int i = acc_sizes_pred.size() - 2; i >= 0; i--) {
			acc_size *= node->operation.shape[i + 2];
			acc_sizes_rest[i] = acc_sizes_rest[i + 1] * slidewin->size[i + 1];
			// no of windows in that dimension
			size_t window_size = pred.shape[i + 1] - slidewin->size[i + 1] + 1;
			window_size = window_size % slidewin->step[i + 1] == 0
							  ? window_size / slidewin->step[i + 1]
							  : window_size / slidewin->step[i + 1] + 1;
			acc_sizes_win[i] = acc_sizes_win[i + 1] * window_size;
		}
		for (size_t i = from; i < from + size; i++) {
			// window number
			size_t wi = i / acc_size;
			size_t rest = i % acc_size;
			// calculate window base from wi
			size_t base = 0;
			// index per dimension inside window from rest
			size_t offset = 0;
			for (int d = 0; d < pred.shape.size(); d++) {
				size_t local_wi = wi / acc_sizes_win[d];
				// top left corner of window in that dimension
				size_t loc_base = local_wi * slidewin->step[d];
				base += loc_base * acc_sizes_pred[d];
				// remove this dimension from wi
				wi %= acc_sizes_win[d];
				size_t local_ri = rest / acc_sizes_rest[d];
				offset += local_ri * acc_sizes_pred[d];
				rest %= acc_sizes_rest[d];
			}
			result[i] = ((const T *__restrict__)data)[base + offset];
		}
	} break;
	case FUNSLIDE_WINDOW: {
		const CPUResultData pred = predecessor_data[0];
		const unsigned int *steps =
			(unsigned int *)node->operation.additional_data;
		const std::vector<size_t> acc_sizes =
			calcAccSizes(node->operation.dimensions, node->operation.shape);
		const std::vector<size_t> acc_sizes_pred =
			calcAccSizes(pred.shape.size(), pred.shape.data());
		size_t no_windows[pred.shape.size() - 1];
		for (int i = 0; i < pred.shape.size() - 1; i++) {
			size_t window_size =
				node->operation.shape[i] - pred.shape[i + 1] + 1;
			no_windows[i] = window_size % steps[i] == 0
								? window_size / steps[i]
								: window_size / steps[i] + 1;
		}
		const std::vector<size_t> acc_no_windows =
			calcAccSizes(pred.shape.size() - 1, no_windows);
		for (size_t i = from; i < from + size; i++) {
			result[i] = 0;
			size_t first_w = 0, last_w = 0;
			// calculate first and last hit
			for (int d = 0; d < node->operation.dimensions; d++) {
				const unsigned int id =
					(i / acc_sizes[d]) % node->operation.shape[d];
				// first hit is where the window overlaps with the element of
				// window size - 1 before this element (since the window reaches
				// to this element)
				const size_t wdf =
					((std::max(0l, (long)id - (long)pred.shape[d + 1] + 1) /
					  steps[d]));
				const size_t wfl = id / steps[d];
				first_w += wdf * acc_no_windows[d];
				last_w += wfl * acc_no_windows[d];
			}
			size_t w = first_w;
			while (w <= last_w) {
				// tests if this window is a hit or not, if not calculates the
				// distance to the next window
				bool contained = true;
				size_t wi = 0;
				size_t wpp = 0;
				for (int d = node->operation.dimensions - 1; d >= 0; d--) {
					const unsigned int wd =
						(w / acc_no_windows[d]) % no_windows[d];
					const unsigned int w_start = wd * steps[d];
					const unsigned int id =
						(i / acc_sizes[d]) % node->operation.shape[d];
					if (id >= w_start && id < w_start + pred.shape[d + 1]) {
						wi += (id - w_start) * acc_sizes_pred[d + 1];
					} else {
						contained = false;
						// we cant break yet -> advance to next window in
						// dimension
						wpp += acc_no_windows[d];
					}
				}
				if (contained) {
					result[i] += ((const T *__restrict__)
									  pred.data)[wi + w * acc_sizes_pred[0]];
					wpp = 1;
				}
				w += wpp;
			}
		}
	} break;
	case FREDUCE_MIN:
	case FREDUCE_MAX:
	case FREDUCE_SUM:
	case FREDUCE_MUL: {
		const CPUResultData pred = predecessor_data[0];
		const int dim = ((int *)node->operation.additional_data)[0];
		const void *__restrict__ data = pred.data;
		size_t it_dim =
			1; // iteration size <=> product of all dimensions along dim
		for (size_t d = dim + 1; d < pred.shape.size(); d++)
			it_dim *= pred.shape[d];

		for (size_t i = from; i < from + size; i++) {
			// iterate through to-reduce dimension
			switch (node->operation.op_type) {
			case FREDUCE_SUM:
				result[i] = 0;
				break;
			case FREDUCE_MUL:
				result[i] = 1;
				break;
			case FREDUCE_MIN:
				result[i] = ((const T *__restrict__)
								 data)[(i / it_dim) * it_dim * pred.shape[dim] +
									   i % it_dim];
				break;
			case FREDUCE_MAX:
				result[i] = ((const T *__restrict__)
								 data)[(i / it_dim) * it_dim * pred.shape[dim] +
									   i % it_dim];
				break;
			default:
				break;
			}
			for (size_t j = 0; j < pred.shape[dim]; j++) {
				const T curr =
					((const T *__restrict__)
						 data)[(i / it_dim) * it_dim * pred.shape[dim] +
							   i % it_dim + j * it_dim];
				switch (node->operation.op_type) {
				case FREDUCE_SUM:
					result[i] += curr;
					break;
				case FREDUCE_MUL:
					result[i] *= curr;
					break;
				case FREDUCE_MIN:
					result[i] = std::min(result[i], curr);
					break;
				case FREDUCE_MAX:
					result[i] = std::max(result[i], curr);
					break;
				default:
					break;
				}
			}
		}
	} break;
	case FSLICE: {
		CPUResultData pred = predecessor_data[0];
		FSlice *slice = (FSlice *)node->operation.additional_data;
		const void *__restrict__ data = pred.data;
		std::vector<size_t> acc_sizes = calcAccSizes(node->operation);
		std::vector<size_t> acc_sizes_pred =
			calcAccSizes(pred.shape.size(), pred.shape.data());
		// calculate start and step size in flattened array
		size_t start = 0;
		for (unsigned int d = 0; d < node->operation.dimensions; d++) {
			start += slice->start[d] * acc_sizes_pred[d];
		}
		// calculate for each entry corresponding element
		for (size_t i = from; i < from + size; i++) {
			size_t j = start;
			for (unsigned int d = 0; d < node->operation.dimensions; d++) {
				// get dimension index
				size_t di = (d == 0 ? i : i % acc_sizes[d - 1]) / acc_sizes[d];
				// reproject
				j += di * slice->step[d] * acc_sizes_pred[d];
			}
			result[i] = ((const T *__restrict__)data)[j];
		}
	} break;
	case FEXTEND: {
		CPUResultData pred = predecessor_data[0];
		const void *__restrict__ data = pred.data;
		FExtend *extend = (FExtend *)node->operation.additional_data;
		std::vector<size_t> acc_sizes = calcAccSizes(node->operation);
		std::vector<size_t> acc_sizes_pred =
			calcAccSizes(pred.shape.size(), pred.shape.data());
		// calculate for each entry corresponding element
		for (size_t i = from; i < from + size; i++) {
			size_t j = 0;
			bool set_zero = false;
			for (size_t d = 0; d < acc_sizes.size(); d++) {
				long step = extend->step[d];
				bool inv = step < 0;
				if (inv)
					step = -step;
				// get dimension index
				size_t di = (d == 0 ? i : i % acc_sizes[d - 1]) / acc_sizes[d];
				if (di < extend->start[d]) {
					set_zero = true;
					break;
				}
				di -= extend->start[d];
				if (di % step != 0) {
					set_zero = true;
					break;
				}
				di /= step;
				if (di >= pred.shape[d]) {
					set_zero = true;
					break;
				}
				// reverse if negative
				if (inv) {
					di = pred.shape[d] - di - 1;
				}
				// reproject
				j += di * acc_sizes_pred[d];
			}
			result[i] = set_zero ? 0 : ((const T *__restrict__)data)[j];
		}
	} break;
	case FABS: {
		CPUResultData pred = predecessor_data[0];
		const void *__restrict__ data = pred.data;
		for (size_t i = from; i < from + size; i++)
			result[i] = abs(((const T *__restrict__)data)[i]);
	} break;
	case FLOG: {
		CPUResultData pred = predecessor_data[0];
		const void *__restrict__ data = pred.data;
		for (size_t i = from; i < from + size; i++) {
			result[i] = log(((const T *__restrict__)data)[i]);
		}
	} break;
	case FLOG2: {
		CPUResultData pred = predecessor_data[0];
		const void *__restrict__ data = pred.data;
		for (size_t i = from; i < from + size; i++)
			result[i] = log2(((const T *__restrict__)data)[i]);
	} break;
	case FNEG: {
		CPUResultData pred = predecessor_data[0];
		const void *__restrict__ data = pred.data;
		for (size_t i = from; i < from + size; i++)
			result[i] = -((const T *__restrict__)data)[i];
	} break;
	case FLOG10: {
		const CPUResultData pred = predecessor_data[0];
		for (size_t i = from; i < from + size; i++)
			result[i] = log10(((const T *__restrict__)pred.data)[i]);
	} break;
	case FSIN: {
		const CPUResultData pred = predecessor_data[0];
		for (size_t i = from; i < from + size; i++)
			result[i] = sin(((const T *__restrict__)pred.data)[i]);
	} break;
	case FSQRT: {
		const CPUResultData pred = predecessor_data[0];
		for (size_t i = from; i < from + size; i++)
			result[i] = sqrt(((const T *__restrict__)pred.data)[i]);
	} break;
	case FEXP: {
		const CPUResultData pred = predecessor_data[0];
		for (size_t i = from; i < from + size; i++)
			result[i] = exp(((const T *__restrict__)pred.data)[i]);
	} break;
	case FCOS: {
		const CPUResultData pred = predecessor_data[0];
		for (size_t i = from; i < from + size; i++)
			result[i] = cos(((const T *__restrict__)pred.data)[i]);
	} break;
	case FTAN: {
		const CPUResultData pred = predecessor_data[0];
		for (size_t i = from; i < from + size; i++)
			result[i] = tan(((const T *__restrict__)pred.data)[i]);
	} break;
	case FASIN: {
		const CPUResultData pred = predecessor_data[0];
		for (size_t i = from; i < from + size; i++)
			result[i] = asin(((const T *__restrict__)pred.data)[i]);
	} break;
	case FACOS: {
		const CPUResultData pred = predecessor_data[0];
		for (size_t i = from; i < from + size; i++)
			result[i] = acos(((const T *__restrict__)pred.data)[i]);
	} break;
	case FATAN: {
		const CPUResultData pred = predecessor_data[0];
		for (size_t i = from; i < from + size; i++)
			result[i] = atan(((const T *__restrict__)pred.data)[i]);
	} break;
	case FSET_INDEX: {
		// TODO Improve (somehow)
		const CPUResultData a = predecessor_data[0];
		const CPUResultData b = predecessor_data[1];
		const CPUResultData c = predecessor_data[2];
		const unsigned int axis = c.shape.size() - 1;
		const FOperation op = node->operation;
		// get index of result, index tensor, reproject index
		size_t acc_sizes_ax = 1;
		for (int i = axis + 1; i < op.dimensions; i++)
			acc_sizes_ax *= op.shape[i];

		for (size_t i = from; i < from + size; i++) {
			const size_t base = i / (acc_sizes_ax * op.shape[axis]);
			const size_t rest = i % acc_sizes_ax;
			const size_t axi = (i / acc_sizes_ax) % op.shape[axis];
			const size_t base_ind = base * c.shape[axis];
			bool found_something = false;
			result[i] = 0;
			// iterate over last dimension and find all correct indices
			for (size_t j = base_ind; j < base_ind + c.shape[axis]; j++) {
				const long ind =
					(long)(c.type == F_INT32 ? ((int *)c.data)[j]
											 : ((long *)c.data)[j]);
				if (ind == axi) {
					found_something = true;
					result[i] += ((T *)b.data)[j * acc_sizes_ax + rest];
				}
			}
			// if at least one index was found -> only sum of elements of b
			if (!found_something)
				result[i] = ((T *)a.data)[i];
		}
	} break;
	default: {
		if (node->num_predecessor == 1) {
			const CPUResultData p1 = predecessor_data[0];
			switch (p1.type) {
			case F_INT32:
				unaryExpression(result, (int *)p1.data, node->operation.op_type,
								from, size, p1.num_entries, node);
				break;
			case F_INT64:
				unaryExpression(result, (long *)p1.data,
								node->operation.op_type, from, size,
								p1.num_entries, node);
				break;
			case F_FLOAT32:
				unaryExpression(result, (float *)p1.data,
								node->operation.op_type, from, size,
								p1.num_entries, node);
				break;
			case F_FLOAT64:
				unaryExpression(result, (double *)p1.data,
								node->operation.op_type, from, size,
								p1.num_entries, node);
				break;
			}
		} else {
			// binary operations
			const CPUResultData p1 = predecessor_data[0],
								p2 = predecessor_data[1];
			size_t im1 = p1.num_entries, im2 = p2.num_entries;
			size_t iv1 = 1, iv2 = 1;
			calculateDivisorForInverseBroadcasting(node->predecessors[0], iv1,
												   node->predecessors[1], iv2);
			switch (p1.type) {
			case F_INT32:
				switch (p2.type) {
				case F_INT32:
					binaryExpression(result, (int *)p1.data, (int *)p2.data,
									 node->operation.op_type, from, size, im1,
									 iv1, im2, iv2, node);
					break;
				case F_FLOAT32:
					binaryExpression(result, (int *)p1.data, (float *)p2.data,
									 node->operation.op_type, from, size, im1,
									 iv1, im2, iv2, node);
					break;
				case F_FLOAT64:
					binaryExpression(result, (int *)p1.data, (double *)p2.data,
									 node->operation.op_type, from, size, im1,
									 iv1, im2, iv2, node);
					break;
				case F_INT64:
					binaryExpression(result, (int *)p1.data, (long *)p2.data,
									 node->operation.op_type, from, size, im1,
									 iv1, im2, iv2, node);
					break;
				}
				break;
			case F_FLOAT32:
				switch (p2.type) {
				case F_INT32:
					binaryExpression(result, (float *)p1.data, (int *)p2.data,
									 node->operation.op_type, from, size, im1,
									 iv1, im2, iv2, node);
					break;
				case F_FLOAT32:
					binaryExpression(result, (float *)p1.data, (float *)p2.data,
									 node->operation.op_type, from, size, im1,
									 iv1, im2, iv2, node);
					break;
				case F_FLOAT64:
					binaryExpression(result, (float *)p1.data,
									 (double *)p2.data, node->operation.op_type,
									 from, size, im1, iv1, im2, iv2, node);
					break;
				case F_INT64:
					binaryExpression(result, (float *)p1.data, (long *)p2.data,
									 node->operation.op_type, from, size, im1,
									 iv1, im2, iv2, node);
					break;
				}
				break;
			case F_FLOAT64:
				switch (p2.type) {
				case F_INT32:
					binaryExpression(result, (double *)p1.data, (int *)p2.data,
									 node->operation.op_type, from, size, im1,
									 iv1, im2, iv2, node);
					break;
				case F_FLOAT32:
					binaryExpression(result, (double *)p1.data,
									 (float *)p2.data, node->operation.op_type,
									 from, size, im1, iv1, im2, iv2, node);
					break;
				case F_FLOAT64:
					binaryExpression(result, (double *)p1.data,
									 (double *)p2.data, node->operation.op_type,
									 from, size, im1, iv1, im2, iv2, node);
					break;
				case F_INT64:
					binaryExpression(result, (double *)p1.data, (long *)p2.data,
									 node->operation.op_type, from, size, im1,
									 iv1, im2, iv2, node);
					break;
				}
				break;
			case F_INT64:
				switch (p2.type) {
				case F_INT32:
					binaryExpression(result, (long *)p1.data, (int *)p2.data,
									 node->operation.op_type, from, size, im1,
									 iv1, im2, iv2, node);
					break;
				case F_FLOAT32:
					binaryExpression(result, (long *)p1.data, (float *)p2.data,
									 node->operation.op_type, from, size, im1,
									 iv1, im2, iv2, node);
					break;
				case F_FLOAT64:
					binaryExpression(result, (long *)p1.data, (double *)p2.data,
									 node->operation.op_type, from, size, im1,
									 iv1, im2, iv2, node);
					break;
				case F_INT64:
					binaryExpression(result, (long *)p1.data, (long *)p2.data,
									 node->operation.op_type, from, size, im1,
									 iv1, im2, iv2, node);
					break;
				}
				break;
			}
			break;
		}
	}
	}
}
#endif
