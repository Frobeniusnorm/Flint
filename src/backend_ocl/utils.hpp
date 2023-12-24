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
static cl_mem push_array(int size, T *data, cl_kernel kernel, cl_context context,
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
static cl_mem calc_and_push_acc_size(int dim, size_t *shape, cl_kernel kernel,
								 cl_context context, int &par_index) {
	std::vector<size_t> acc_sizes(dim);
	acc_sizes[dim - 1] = 1;
	for (long d = dim - 2; d >= 0; d--) {
		acc_sizes[d] = acc_sizes[d + 1] * shape[d + 1];
	}
	return push_array(acc_sizes.size(), acc_sizes.data(), kernel, context,
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
#endif
