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

#ifndef GRADIENTS_CPP
#define GRADIENTS_CPP
#include "../flint.h"
#include "../flint_helper.hpp"
#include "backend_ocl/comp.hpp"
#include "src/errors.hpp"
#include "src/operations/implementation.hpp"
#include "utils.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <list>
#include <math.h>
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#define MIN_VAL(x, y) (x) < (y) ? (x) : (y)
#define MAX_VAL(x, y) (x) < (y) ? (y) : (x)
static inline void
configureGradientInformation(FGraphNode *g, std::vector<FGraphNode *> pred) {
	std::unordered_set<const FGraphNode *> *gd = nullptr;
	for (FGraphNode *p : pred) {
		if (p->gradient_data) {
			if (!gd)
				gd = new std::unordered_set<const FGraphNode *>();
			std::unordered_set<const FGraphNode *> *other =
				(std::unordered_set<const FGraphNode *> *)p->gradient_data;
			gd->reserve(other->size() + gd->size());
			for (const FGraphNode *g : *other) {
				// check if it is still a variable
				if (g->gradient_data)
					gd->insert(g);
			}
		}
	}
	g->gradient_data = (void *)gd;
}
static FGraphNode *constant_tensor(double val, FType type, size_t *shape,
								   int dimensions) {
	switch (type) {
	case F_FLOAT32:
		return fconstant_f((float)val, shape, dimensions);
	case F_INT32:
		return fconstant_i((int)val, shape, dimensions);
	case F_INT64:
		return fconstant_l((long)val, shape, dimensions);
	case F_FLOAT64:
		return fconstant_d((double)val, shape, dimensions);
	}
}
static FGraphNode *unbroadcast(FGraphNode *adjoint, const FGraphNode *node) {
	if (adjoint->operation.dimensions > node->operation.dimensions) {
		size_t diff =
			adjoint->operation.dimensions - node->operation.dimensions;
		FGraphNode *res = adjoint;
		for (int i = 0; i < diff; i++) {
			res = freduce_sum(res, 0);
		}
		return res;
	} else if (adjoint->operation.dimensions < node->operation.dimensions) {
		size_t diff =
			node->operation.dimensions - adjoint->operation.dimensions;
		std::vector<size_t> new_shape(node->operation.dimensions);
		std::vector<int> repetitions(node->operation.dimensions, 0);
		for (int i = 0; i < diff; i++) {
			new_shape[i] = 1;
			repetitions[i] = node->operation.shape[i] - 1;
		}
		for (int i = diff; i < new_shape.size(); i++)
			new_shape[i] = adjoint->operation.shape[i - diff];
		FGraphNode *res = freshape(adjoint, new_shape.data(), new_shape.size());
		res = frepeat(res, repetitions.data());
		return res;
	}
	return adjoint;
}
static void collect(FGraphNode *x, std::list<FGraphNode *> &stack,
					std::unordered_set<FGraphNode *> &visited,
					const std::unordered_set<const FGraphNode *> dxs) {
	// TODO could be made more performant with explicit todo stack and a
	// push_back before continuing on the parents, see utils
	if (visited.contains(x))
		return;
	visited.insert(x);
	for (int i = 0; i < x->num_predecessor; i++) {
		FGraphNode *parent = x->predecessors[i];
		// check if visited
		if (visited.contains(parent))
			continue;
		// check if it contains dx
		if (parent->gradient_data) {
			std::unordered_set<const FGraphNode *> *trace =
				(std::unordered_set<const FGraphNode *> *)parent->gradient_data;
			bool skip = true;
			for (const FGraphNode *dx : dxs) {
				if (trace->contains(dx)) {
					skip = false;
					break;
				}
			}
			if (skip)
				continue;
		} else if (!dxs.contains(parent))
			continue;
		// recurse
		collect(parent, stack, visited, dxs);
	}
	stack.push_front(x);
}
FGraphNode *fCalculateGradient(FGraphNode *y, FGraphNode *dx) {
	FGraphNode *res;
	fCalculateGradients(y, &dx, 1, &res);
	return res;
}
FErrorType fCalculateGradients(FGraphNode *y, FGraphNode **dx,
							   const unsigned int num_gradients,
							   FGraphNode **gradients) {
	using namespace std;
	// cout << " = Calculate Gradients = " << endl;
	unordered_set<const FGraphNode *> *gd =
		(unordered_set<const FGraphNode *> *)y->gradient_data;
	if (!gd) {
		setErrorType(ILLEGAL_DERIVE);
		flogging(
			F_ERROR,
			"no derivatives in the operational graph! Don't forget the "
			"necessary calls to fMarkGradientVariable (or in C++ .watch())");
		return ILLEGAL_DERIVE;
	}
	std::unordered_set<const FGraphNode *> vars(num_gradients);
	for (int i = 0; i < num_gradients; i++) {
		vars.insert(dx[i]);
		if (!gd->contains(dx[i]))
			flogging(
				F_WARNING,
				"derivative was not marked during graph construction! Don't "
				"forget the "
				"necessary calls to fMarkGradientVariable (or in C++ "
				".watch())");
	}
	// to store gradients per node
	unordered_map<const FGraphNode *, FGraphNode *> adjoints;
	list<FGraphNode *> todo;
	std::unordered_set<FGraphNode *> visited;
	collect(y, todo, visited, vars);
	// used to determine when a node may be freed
	unordered_map<const FGraphNode *, unsigned int> needed_by(adjoints.size());
	// initialize
	adjoints[y] = constant_tensor(1., y->operation.data_type,
								  y->operation.shape, y->operation.dimensions);
	for (FGraphNode *curr : todo) {
		FGraphNode *adj = adjoints[curr];
		bool allowed_to_free = true;
		adj->reference_counter++;
		for (int i = 0; i < curr->num_predecessor; i++) {
			FGraphNode *parent = curr->predecessors[i];
			if (!visited.contains(parent))
				continue;
			auto start = std::chrono::high_resolution_clock::now();
			FGraphNode *local_grad = unbroadcast(
				OperationImplementation::implementations[curr->operation
															 .op_type]
					->local_gradient(curr, i, adj),
				parent);
			if (adjoints.contains(parent)) {
				adjoints[parent] =
					fExecuteGraph(fadd(adjoints[parent], local_grad));
			} else {
				adjoints.insert({parent, fExecuteGraph(local_grad)});
				if (local_grad == adj)
					allowed_to_free = false;
			}
			fOptimizeMemory(adjoints[parent]);
			std::chrono::duration<double, std::milli> elapsed =
				std::chrono::high_resolution_clock::now() - start;
			// std::cout << fop_to_string[curr->operation.op_type] << " took "
			//           << elapsed.count() << " for " << i << " type: " <<
			//           type_string(local_grad->operation.data_type) <<
			//           std::endl;
		}
		if (!vars.contains(curr)) {
			if (--adj->reference_counter <= 0) {
				if (allowed_to_free) {
					fFreeGraph(adj);
				}
			}
			adjoints[curr] = nullptr;
		}
	}
	for (const FGraphNode *v : vars)
		adjoints[v]->reference_counter--;
	for (int i = 0; i < num_gradients; i++) {
		if (adjoints.contains(dx[i])) {
			gradients[i] = adjoints[dx[i]];
			FType higher =
				higher_type(y->operation.data_type, dx[i]->operation.data_type);
			if (higher == F_INT32 || higher == F_INT64)
				higher = F_FLOAT64;
			if (gradients[i]->operation.data_type != higher)
				gradients[i] = fconvert(gradients[i], higher);
		} else {
			flogging(F_WARNING,
					 "Operation graph did not contain the derivative!");
			gradients[i] = nullptr;
		}
	}
	return NO_ERROR;
}
#endif
