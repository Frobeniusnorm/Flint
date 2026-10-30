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

#ifndef OCL_CODEGEN_HPP
#define OCL_CODEGEN_HPP
#include <vector>
#define FLINT_DEBUG
#include "../../flint.h"
#include <list>
#include <string>
/** OpenCL range of a kernel, innermost dimension first */
struct LaunchRange {
		int dims = 1;
		size_t size[3] = {1, 1, 1};
};
/** OpenCL has only three range dimensions. Axes of size 1 are dropped since
 * they have no stride, the two innermost remaining ones get a dimension each
 * and all outer ones are folded into the third. Only neighbouring axes are
 * folded, so each dimension stays a contiguous block of the tensor and the
 * index of an element is a plain sum of strided ids */
inline LaunchRange launchRange(const FGraphNode *node) {
	LaunchRange range;
	if (node->operation.op_type == FGEN_CONSTANT)
		return range;
	int found = 0;
	for (int i = node->operation.dimensions - 1; i >= 0; i--) {
		const size_t size = node->operation.shape[i];
		if (size == 1)
			continue;
		if (found < 3)
			range.size[found++] = size;
		else
			range.size[2] *= size;
	}
	range.dims = found ? found : 1;
	return range;
}
/**
 * Generates OpenCL code for an OpenCL node (Only the calculation for the graph!
 * Nothing else). This function is called by `fExecuteGraph_GPU`.
 * - `node`: The result of the calculation. The OpenCL kernel contains
 *   its calculation.
 * - `parameters`: Node and name of the paramters to the kernel. This functions
 *   fills this list. It finds the parameters as the nodes with results .
 * - `scalars`: All parameters that are constant scalars.
 */
std::string
generateCode(FGraphNode *node,
			 std::list<std::pair<FGraphNode *, std::string>> &parameters,
			 std::vector<std::pair<FType, double>> &scalars);
#endif
