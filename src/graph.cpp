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

#include "../flint.h"
#include "backend_ocl/comp.hpp"
#include "errors.hpp"
#include "src/operations/implementation.hpp"
#include "utils.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <list>
#include <stdlib.h>
#include <string>
#include <unordered_set>
#include <vector>
#define MAX(x, y) (x) > (y) ? (x) : (y)
#define ABS(x) (x) < 0 ? -(x) : (x)
const char *fop_to_string[] = {"FSTORE",
							   "FGEN_RANDOM",
							   "FGEN_CONST",
							   "FGEN_ARANGE",
							   "FADD",
							   "FSUB",
							   "FMUL",
							   "FDIV",
							   "FPOW",
							   "FNEG",
							   "FLOG",
							   "FSIGN",
							   "FEVEN",
							   "FLOG2",
							   "FLOG10",
							   "FSIN",
							   "FCOS",
							   "FTAN",
							   "FASIN",
							   "FACOS",
							   "FATAN",
							   "FSQRT",
							   "FEXP",
							   "FLATTEN",
							   "FMATMUL",
							   "FCONVERSION",
							   "FRESHAPE",
							   "FMIN",
							   "FMAX",
							   "FREDUCE_SUM",
							   "FREDUCE_MUL",
							   "FREDUCE_MIN",
							   "FREDUCE_MAX",
							   "FSLICE",
							   "FABS",
							   "FREPEAT",
							   "FTRANSPOSE",
							   "FEXTEND",
							   "FCONCAT",
							   "FLESS",
							   "FEQUAL",
							   "FGREATER",
							   "FCONVOLVE",
							   "FGRADIENT_CONVOLVE1",
							   "FGRADIENT_CONVOLVE2",
							   "FINDEX",
							   "FSET_INDEX",
							   "FSLIDING_WINDOW",
							   "FUNSLIDE_WINDOW",
							   "FPOOLING_MAX",
							   "FPOOLING_SUM",
							   "FGRADIENT_POOLING_MAX",
							   "FDROPOUT"};
static bool use_cpu, use_gpu, gradient_context = false;
static FErrorType last_error;
void setErrorType(FErrorType error) { last_error = error; }
// TODO do execution of parents where necessary in parallel
// EAGER EXECUTION WITH HELPER
void fStartGradientContext() { gradient_context = true; }
void fStopGradientContext() { gradient_context = false; }
bool fIsGradientContext() { return gradient_context; }
FErrorType fErrorType() { return last_error; }
static inline void
configureGradientInformation(FGraphNode *g, std::vector<FGraphNode *> pred) {
	if (!gradient_context)
		return;
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
static const int cores = std::thread::hardware_concurrency();
// INTERFACE METHODS
FGraphNode *fExecuteGraph(FGraphNode *node) {
	if (!use_cpu && !use_gpu)
		if (flintInit(FLINT_BACKEND_BOTH) != NO_ERROR)
			return nullptr;
	if (use_gpu && use_cpu) {
		size_t no_elems = 1;
		for (int i = 0; i < node->operation.dimensions; i++)
			no_elems *= node->operation.shape[i];
		const unsigned int gpu_score = compute_score(node, true);
		int cpu_boost = 2, gpu_boost = 2;
		for (int i = 0; i < node->num_predecessor; i++) {
			if (node->result_data) {
				if (!node->result_data->data)
					cpu_boost = 1;
				if (!node->result_data->mem_id)
					gpu_boost = 1;
			}
		}
		return no_elems * gpu_score * gpu_boost / cpu_boost >= 1024
				   ? fExecuteGraph_gpu(node)
				   : fExecuteGraph_cpu(node);
	}
	if (use_gpu)
		return fExecuteGraph_gpu(node);
	if (use_cpu)
		return fExecuteGraph_cpu(node);
	return nullptr;
}
FGraphNode *fCalculateResult(FGraphNode *node) {
	node = fExecuteGraph(node);
	fSyncMemory(node);
	return node;
}
FErrorType flintCleanup() {
	// for (OperationImplementation *impl :
	// 	 OperationImplementation::implementations)
	// 	delete impl;
	FErrorType e1 = flintCleanup_cpu();
	if (e1 != NO_ERROR)
		return e1;
	FErrorType e2 = flintCleanup_gpu();
	if (e2 != NO_ERROR)
		return e2;
	use_cpu = false;
	use_gpu = false;
	return NO_ERROR;
}
FErrorType flintInit(int backends) {
	flogging(F_VERBOSE, "Initializing Flint");
	std::srand((unsigned int)(std::time(nullptr)));
	use_cpu = (backends & FLINT_BACKEND_ONLY_CPU);
	use_gpu = (backends & FLINT_BACKEND_ONLY_GPU);
	FErrorType e1 = NO_ERROR, e2 = NO_ERROR;
	if (use_cpu)
		e1 = flintInit_cpu();
	if (use_gpu)
		e2 = flintInit_gpu();
	if (e1 != NO_ERROR)
		return e1;
	if (e2 != NO_ERROR)
		return e2;
	return NO_ERROR;
}
int flintInitializedBackends() {
	int backends = 0;
	if (use_cpu)
		backends |= FLINT_BACKEND_ONLY_CPU;
	if (use_gpu)
		backends |= FLINT_BACKEND_ONLY_GPU;
	return backends;
}
// GRAPH METHODS
FGraphNode *fCreateGraph(const void *data, const int num_entries,
						 const FType data_type, const size_t *shape,
						 const int dimensions) {
	FGraphNode *gn = new FGraphNode();
	gn->gradient_data = nullptr;
	gn->reference_counter = 0;
	gn->result_data = nullptr;
	FOperation op;
	op.broadcasting_mode = 0;
	FStore *store = new FStore();
	store->mem_id = nullptr;
	op.dimensions = dimensions;
	op.shape = safe_mal<size_t>(dimensions);
	if (!op.shape)
		return nullptr;
	std::memcpy((void *)op.shape, (void *)shape, dimensions * sizeof(size_t));
	op.additional_data = (void *)store;
	op.op_type = FSTORE;
	size_t byte_size = num_entries;
	switch (data_type) {
	case F_INT32:
		store->data = safe_mal<int>(num_entries);
		if (!store->data)
			return nullptr;
		byte_size *= sizeof(int);
		break;
	case F_INT64:
		store->data = safe_mal<long>(num_entries);
		if (!store->data)
			return nullptr;
		byte_size *= sizeof(long);
		break;
	case F_FLOAT32:
		store->data = safe_mal<float>(num_entries);
		if (!store->data)
			return nullptr;
		byte_size *= sizeof(float);
		break;
	case F_FLOAT64:
		store->data = safe_mal<long>(num_entries);
		if (!store->data)
			return nullptr;
		byte_size *= sizeof(double);
		break;
	}
	memcpy(store->data, data, byte_size);
	store->num_entries = num_entries;
	op.data_type = data_type;
	gn->operation = op;
	gn->num_predecessor = 0;
	gn->predecessors = NULL;
	return gn;
}
// frees all allocated data from the graph and the nodes that are reachable
void fFreeGraph(FGraphNode *graph) {
	if (!use_cpu && !use_gpu)
		flogging(F_WARNING,
				 "freeing data with no active backend may lead to "
				 "undefined behaviour (maybe you did not initialize any "
				 "backend or already called flintCleanup())!");
	std::unordered_set<const FGraphNode *>
		all; // all which are in the queue and were visited
	std::list<FGraphNode *> wq;
	all.insert(graph);
	wq.push_back(graph);
	OCLCompilerThread::memory_barrier();
	while (!wq.empty()) {
		FGraphNode *gn = wq.front();
		wq.pop_front();
		if (gn->reference_counter > 0) {
			continue;
		}
		for (int i = 0; i < gn->num_predecessor; i++) {
			if (gn->predecessors[i] &&
				--(gn->predecessors[i]->reference_counter) == 0 &&
				all.find(gn->predecessors[i]) == all.end()) {
				wq.push_back(gn->predecessors[i]);
				all.insert(gn->predecessors[i]);
			}
		}
		if (gn->gradient_data) {
			delete (std::unordered_set<const FGraphNode *> *)gn->gradient_data;
		}
		bool freed_res = false;
		if (gn->result_data != nullptr) {
			freed_res = true;
			FResultData *rd = gn->result_data;
			if (rd->data)
				free(rd->data);
			if (rd->mem_id)
				clReleaseMemObject(rd->mem_id);
			rd->mem_id = nullptr;
			delete gn->result_data;
			gn->result_data = nullptr;
		}
		if (gn->predecessors != NULL && gn->num_predecessor != 0)
			free(gn->predecessors);
		if (gn->operation.shape)
			free(gn->operation.shape);
		if (gn->operation.additional_data) {
			switch (gn->operation.op_type) {
			case FSTORE: {
				FStore *st = (FStore *)gn->operation.additional_data;
				if (!freed_res) {
					free(st->data);
					if (st->mem_id) {
						clReleaseMemObject(st->mem_id);
						st->mem_id = nullptr;
					}
				}
				delete st;
			} break;
			default:
				OperationImplementation::implementations[gn->operation.op_type]
					->free_additional_data(gn);
			}
			gn->operation.additional_data = nullptr;
		}
		delete gn;
	}
}
// function to add nodes to the graph i.e. operations
static FGraphNode *addNode(FOperation op, std::vector<FGraphNode *> pre) {
	FGraphNode *foo = new FGraphNode();
	configureGradientInformation(foo, pre);
	foo->reference_counter = 0;
	foo->operation = op;
	foo->result_data = nullptr;
	foo->num_predecessor = pre.size();
	foo->predecessors =
		pre.size() == 0 ? NULL : safe_mal<FGraphNode *>(pre.size());
	if (pre.size() != 0 && !foo->predecessors)
		return nullptr;
	for (size_t i = 0; i < pre.size(); i++) {
		foo->predecessors[i] = pre[i];
		if (pre[i]->reference_counter++ > 2)
			fExecuteGraph(pre[i]);
	}
	return foo;
}
static inline void initShape_keep(FOperation &op, const FOperation *a,
								  const FOperation *b) {
	size_t *src = nullptr;
	size_t *lower = nullptr;
	int lower_dim = -1;
	int broadcasting_mode = 0;
	if (!b || a->dimensions >= b->dimensions) {
		op.dimensions = a->dimensions;
		src = a->shape;
		if (b) {
			lower = b->shape;
			lower_dim = b->dimensions;
			if (b->broadcasting_mode != 0)
				broadcasting_mode = b->broadcasting_mode + 1;
			if (a->dimensions == b->dimensions && src[0] == 1) {
				lower = a->shape;
				src = b->shape;
			}
		}
	} else {
		op.dimensions = b->dimensions;
		src = b->shape;
		lower = a->shape;
		lower_dim = a->dimensions;
		if (a->broadcasting_mode != 0)
			broadcasting_mode = a->broadcasting_mode + 1;
	}
	// check shape if both are defined and lower is not a constant
	if (lower && !(lower_dim == 1 && lower[0] == 1)) {
		for (int i = 0; i < lower_dim; i++) {
			const size_t s1 = src[i + (op.dimensions - lower_dim)];
			const size_t s2 = lower[i];
			const size_t s3 = src[i];
			if (broadcasting_mode == 0) {
				if (s1 == s2 && s2 != s3)
					broadcasting_mode = 1;
				if (s2 == s3 && s1 != s2)
					broadcasting_mode = 2;
			}
			if (broadcasting_mode == 2 ? s2 != s3 : s1 != s2)
				flogging(F_ERROR, "incompatible shapes of operands: " +
									  vector_string(std::vector<size_t>(
										  src, src + op.dimensions)) +
									  " and " +
									  vector_string(std::vector<size_t>(
										  lower, lower + lower_dim)) +
									  " in " + fop_to_string[op.op_type]);
		}
	}
	op.broadcasting_mode = broadcasting_mode == 2 ? 1 : 0;
	op.shape = (size_t *)malloc(sizeof(size_t) * op.dimensions);
	memcpy((void *)op.shape, src, sizeof(size_t) * op.dimensions);
	// determine type
	op.data_type = b ? higher_type(a->data_type, b->data_type) : a->data_type;
}
void fEnforceInverseBroadcasting(FGraphNode *node) {
	node->operation.broadcasting_mode = 1;
}
void fUnenforceInverseBroadcasting(FGraphNode *node) {
	node->operation.broadcasting_mode = 0;
}
void fMarkGradientVariable(FGraphNode *node) {
	std::unordered_set<const FGraphNode *> *trace =
		node->gradient_data
			? (std::unordered_set<const FGraphNode *> *)node->gradient_data
			: new std::unordered_set<const FGraphNode *>();
	if (node->gradient_data && trace->contains(node))
		return;
	trace->insert(node);
	node->gradient_data = (void *)trace;
}
void fUnmarkGradientVariable(FGraphNode *node) {
	if (node->gradient_data) {
		std::unordered_set<const FGraphNode *> *gd =
			(std::unordered_set<const FGraphNode *> *)node->gradient_data;
		gd->erase(node);
		if (gd->empty()) {
			delete gd;
			node->gradient_data = nullptr;
		}
	}
}
FGraphNode *fOptimizeMemory(FGraphNode *node) {
	if (!node->gradient_data && node->operation.op_type != FSTORE &&
		node->operation.op_type != FGEN_CONSTANT && node->result_data) {
		FResultData *rd = node->result_data;
		// we can modify this node to a STORE operation
		OperationImplementation::implementations[node->operation.op_type]
			->free_additional_data(node);
		node->operation.op_type = FSTORE;
		if (flintInitializedBackends() & FLINT_BACKEND_ONLY_GPU) {
			// we can do this only when all operations have been finished
			OCLCompilerThread::memory_barrier();
		}
		for (int i = 0; i < node->num_predecessor; i++) {
			if (--node->predecessors[i]->reference_counter == 0) {
				fFreeGraph(node->predecessors[i]);
			}
		}
		node->num_predecessor = 0;
		free(node->predecessors);
		node->predecessors = nullptr;
		FStore *store = new FStore();
		store->data = rd->data;
		store->mem_id = rd->mem_id;
		store->num_entries = rd->num_entries;
		node->operation.additional_data = store;
	} else if (node->gradient_data && node->result_data) {
		// if the result data of the parent is not needed for certain gradient
		// calculation operations, it may be freed
		switch (node->operation.op_type) {
		case FADD:
		case FNEG:
		case FCONCAT:
		case FSUB:
		case FLATTEN:
		case FRESHAPE:
		case FSLIDING_WINDOW:
		case FTRANSPOSE:
		case FCONVERSION:
		case FREDUCE_SUM:
		case FREDUCE_MUL:
		case FREPEAT:
		case FSLICE:
		case FEXTEND:
		case FSIGN:
		case FEVEN:
		case FLESS:
		case FEQUAL:
		case FGREATER:
			// all parents that are only referenced by this node can be freed
			for (int i = 0; i < node->num_predecessor; i++) {
				FGraphNode *parent = node->predecessors[i];
				if (parent->result_data && parent->reference_counter <= 2 &&
					parent->operation.op_type != FSTORE) {
					FResultData *rd = parent->result_data;
					if (rd->data)
						free(rd->data);
					if (rd->mem_id)
						clReleaseMemObject(rd->mem_id);
					rd->mem_id = nullptr;
					delete rd;
					parent->result_data = nullptr;
				}
			}
		default:
			break;
		}
	}
	return node;
}
FGraphNode *fadd_g(FGraphNode *a, FGraphNode *b) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FADD;
	initShape_keep(op, &a->operation, &b->operation);
	op.data_type = higher_type(a->operation.data_type, b->operation.data_type);
	return addNode(op, {a, b});
}
FGraphNode *fsub_g(FGraphNode *a, FGraphNode *b) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FSUB;
	initShape_keep(op, &a->operation, &b->operation);
	op.data_type = higher_type(a->operation.data_type, b->operation.data_type);
	return addNode(op, {a, b});
}
FGraphNode *fdiv_g(FGraphNode *a, FGraphNode *b) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FDIV;
	initShape_keep(op, &a->operation, &b->operation);
	op.data_type = higher_type(a->operation.data_type, b->operation.data_type);
	return addNode(op, {a, b});
}
FGraphNode *fmul_g(FGraphNode *a, FGraphNode *b) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FMUL;
	initShape_keep(op, &a->operation, &b->operation);
	op.data_type = higher_type(a->operation.data_type, b->operation.data_type);
	return addNode(op, {a, b});
}
FGraphNode *fpow_g(FGraphNode *a, FGraphNode *b) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FPOW;
	initShape_keep(op, &a->operation, &b->operation);
	op.data_type = higher_type(a->operation.data_type, b->operation.data_type);
	return addNode(op, {a, b});
}
FGraphNode *fmin_g(FGraphNode *a, FGraphNode *b) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FMIN;
	initShape_keep(op, &a->operation, &b->operation);
	op.data_type = higher_type(a->operation.data_type, b->operation.data_type);
	return addNode(op, {a, b});
}
FGraphNode *fmax_g(FGraphNode *a, FGraphNode *b) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FMAX;
	initShape_keep(op, &a->operation, &b->operation);
	op.data_type = higher_type(a->operation.data_type, b->operation.data_type);
	return addNode(op, {a, b});
}
// creates tensor consisting of a single value
template <typename T>
static inline FGraphNode *constant(const T value, const size_t *shape,
								   const int dimensions) {
	FOperation op;
	op.broadcasting_mode = 0;
	op.dimensions = dimensions;
	op.shape = safe_mal<size_t>(dimensions);
	if (!op.shape)
		return nullptr;
	memcpy(op.shape, shape, op.dimensions * sizeof(size_t));
	op.op_type = FGEN_CONSTANT;
	op.data_type = to_flint_type<T>();
	op.additional_data = safe_mal<T>(1);
	if (!op.additional_data)
		return nullptr;
	((T *)op.additional_data)[0] = value;
	return addNode(op, {});
}
template <typename T>
static FGraphNode *addNodeWithConst(FOperation op, FGraphNode *a, const T b) {
	return addNode(
		op, {a, constant(b, a->operation.shape, a->operation.dimensions)});
}
template <typename T>
static FGraphNode *addConstWithNode(FOperation op, const T b, FGraphNode *a) {
	return addNode(
		op, {constant(b, a->operation.shape, a->operation.dimensions), a});
}
FGraphNode *fconstant_i(const int value, const size_t *shape,
						const int dimensions) {
	return constant(value, shape, dimensions);
}
FGraphNode *fconstant_l(const long value, const size_t *shape,
						const int dimensions) {
	return constant(value, shape, dimensions);
}

FGraphNode *fconstant_f(const float value, const size_t *shape,
						const int dimensions) {
	return constant(value, shape, dimensions);
}

FGraphNode *fconstant_d(const double value, const size_t *shape,
						const int dimensions) {
	return constant(value, shape, dimensions);
}

FGraphNode *farange(const size_t *shape, const int dimensions, const int ax) {
	FOperation op;
	op.broadcasting_mode = 0;
	op.dimensions = dimensions;
	op.shape = safe_mal<size_t>(dimensions);
	if (!op.shape)
		return nullptr;
	memcpy(op.shape, shape, op.dimensions * sizeof(size_t));
	op.op_type = FGEN_ARANGE;
	op.data_type = F_INT64;
	op.additional_data = safe_mal<int>(1);
	if (!op.additional_data)
		return nullptr;
	((int *)op.additional_data)[0] = ax;
	return addNode(op, {});
}
// adds the constant value to each entry in a
template <typename T> static inline FGraphNode *add(FGraphNode *a, const T b) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FADD;
	initShape_keep(op, &a->operation, nullptr);
	op.data_type = higher_type(a->operation.data_type, to_flint_type<T>());
	FGraphNode *foo = addNodeWithConst(op, a, b);
	return foo;
}
FGraphNode *fadd_cd(FGraphNode *a, const double b) { return add<double>(a, b); }
FGraphNode *fadd_cf(FGraphNode *a, const float b) { return add<float>(a, b); }
FGraphNode *fadd_ci(FGraphNode *a, const int b) { return add<int>(a, b); }
FGraphNode *fadd_cl(FGraphNode *a, const long b) { return add<long>(a, b); }
// subtracts the constant value from each entry in a
template <typename T> static inline FGraphNode *sub(FGraphNode *a, const T b) {
	FOperation op;
	op.op_type = FSUB;
	op.additional_data = nullptr;
	initShape_keep(op, &a->operation, nullptr);
	op.data_type = higher_type(a->operation.data_type, to_flint_type<T>());
	return addNodeWithConst(op, a, b);
}
template <typename T> static inline FGraphNode *sub(const T b, FGraphNode *a) {
	FOperation op;
	op.op_type = FSUB;
	op.additional_data = nullptr;
	initShape_keep(op, &a->operation, nullptr);
	return addConstWithNode(op, b, a);
}
FGraphNode *fsub_cd(FGraphNode *a, const double b) { return sub<double>(a, b); }
FGraphNode *fsub_cf(FGraphNode *a, const float b) { return sub<float>(a, b); }
FGraphNode *fsub_ci(FGraphNode *a, const int b) { return sub<int>(a, b); }
FGraphNode *fsub_cl(FGraphNode *a, const long b) { return sub<long>(a, b); }

FGraphNode *fsub_icd(const double b, FGraphNode *a) {
	return sub<double>(b, a);
}
FGraphNode *fsub_icf(const float b, FGraphNode *a) { return sub<float>(b, a); }
FGraphNode *fsub_ici(const int b, FGraphNode *a) { return sub<int>(b, a); }
FGraphNode *fsub_icl(const long b, FGraphNode *a) { return sub<long>(b, a); }
// divides each entry in a by the constant value
template <typename T> static inline FGraphNode *div(FGraphNode *a, const T b) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FDIV;
	initShape_keep(op, &a->operation, nullptr);
	op.data_type = higher_type(a->operation.data_type, to_flint_type<T>());
	return addNodeWithConst(op, a, b);
}
template <typename T> static inline FGraphNode *div(const T b, FGraphNode *a) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FDIV;
	initShape_keep(op, &a->operation, nullptr);
	return addConstWithNode(op, b, a);
}
FGraphNode *fdiv_cd(FGraphNode *a, const double b) { return div<double>(a, b); }
FGraphNode *fdiv_cf(FGraphNode *a, const float b) { return div<float>(a, b); }
FGraphNode *fdiv_ci(FGraphNode *a, const int b) { return div<int>(a, b); }
FGraphNode *fdiv_cl(FGraphNode *a, const long b) { return div<long>(a, b); }

FGraphNode *fdiv_icd(const double b, FGraphNode *a) {
	return div<double>(b, a);
}
FGraphNode *fdiv_icf(const float b, FGraphNode *a) { return div<float>(b, a); }
FGraphNode *fdiv_ici(const int b, FGraphNode *a) { return div<int>(b, a); }
FGraphNode *fdiv_icl(const long b, FGraphNode *a) { return div<long>(b, a); }
// multiplicates the constant value with each entry in a
template <typename T> static inline FGraphNode *mul(FGraphNode *a, const T b) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FMUL;
	initShape_keep(op, &a->operation, nullptr);
	op.data_type = higher_type(a->operation.data_type, to_flint_type<T>());
	return addNodeWithConst(op, a, b);
}
FGraphNode *fmul_cd(FGraphNode *a, const double b) { return mul<double>(a, b); }
FGraphNode *fmul_cf(FGraphNode *a, const float b) { return mul<float>(a, b); }
FGraphNode *fmul_ci(FGraphNode *a, const int b) { return mul<int>(a, b); }
FGraphNode *fmul_cl(FGraphNode *a, const long b) { return mul<long>(a, b); }
// takes the power of each element in a to b
template <typename T> static inline FGraphNode *pow(FGraphNode *a, const T b) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FPOW;
	initShape_keep(op, &a->operation, nullptr);
	op.data_type = higher_type(a->operation.data_type, to_flint_type<T>());
	return addNodeWithConst(op, a, b);
}
FGraphNode *fpow_cd(FGraphNode *a, const double b) { return pow<double>(a, b); }
FGraphNode *fpow_cf(FGraphNode *a, const float b) { return pow<float>(a, b); }
FGraphNode *fpow_ci(FGraphNode *a, const int b) { return pow<int>(a, b); }
FGraphNode *fpow_cl(FGraphNode *a, const long b) { return pow<long>(a, b); }

template <typename T> static inline FGraphNode *min(FGraphNode *a, const T b) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FMIN;
	initShape_keep(op, &a->operation, nullptr);
	op.data_type = higher_type(a->operation.data_type, to_flint_type<T>());
	return addNodeWithConst(op, a, b);
}
FGraphNode *fmin_ci(FGraphNode *a, const int b) { return min(a, b); }
FGraphNode *fmin_cl(FGraphNode *a, const long b) { return min(a, b); }
FGraphNode *fmin_cf(FGraphNode *a, const float b) { return min(a, b); }
FGraphNode *fmin_cd(FGraphNode *a, const double b) { return min(a, b); }

template <typename T> static inline FGraphNode *max(FGraphNode *a, const T b) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FMAX;
	initShape_keep(op, &a->operation, nullptr);
	op.data_type = higher_type(a->operation.data_type, to_flint_type<T>());
	return addNodeWithConst(op, a, b);
}
FGraphNode *fmax_ci(FGraphNode *a, const int b) { return max(a, b); }
FGraphNode *fmax_cl(FGraphNode *a, const long b) { return max(a, b); }
FGraphNode *fmax_cf(FGraphNode *a, const float b) { return max(a, b); }
FGraphNode *fmax_cd(FGraphNode *a, const double b) { return max(a, b); }

static inline FGraphNode *log_impl(FGraphNode *a,
								   const FOperationType logtype) {
	FOperation op;
	op.broadcasting_mode = 0;
	op.op_type = logtype;
	op.dimensions = a->operation.dimensions;
	op.shape = safe_mal<size_t>(op.dimensions * sizeof(size_t));
	if (!op.shape)
		return nullptr;
	op.additional_data = nullptr;
	memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
	op.data_type = a->operation.data_type;
	if (op.data_type == F_INT32 || op.data_type == F_INT64) {
		a = fconvert(a, F_FLOAT64);
		op.data_type = F_FLOAT64;
	}
	return addNode(op, {a});
}
/** Takes the elementwise natural logarithm of a */
FGraphNode *flog(FGraphNode *a) { return log_impl(a, FLOG); }
/** Takes the elementwise logarithm of a to the basis of 2*/
FGraphNode *flog2(FGraphNode *a) { return log_impl(a, FLOG2); }
/** Takes the elementwise logarithm of a to the basis of 10*/
FGraphNode *flog10(FGraphNode *a) { return log_impl(a, FLOG10); }
/** Takes the elementwise sinus of a */
FGraphNode *fsin(FGraphNode *a) { return log_impl(a, FSIN); }
/** Takes the elementwise cosinus of a */
FGraphNode *fcos(FGraphNode *a) { return log_impl(a, FCOS); }
/** Takes the elementwise tangents of a */
FGraphNode *ftan(FGraphNode *a) { return log_impl(a, FTAN); }
/** Takes the elementwise inverse sinus of a */
FGraphNode *fasin(FGraphNode *a) { return log_impl(a, FASIN); }
/** Takes the elementwise inverse cosinus of a */
FGraphNode *facos(FGraphNode *a) { return log_impl(a, FACOS); }
/** Takes the elementwise inverse tangents of a */
FGraphNode *fatan(FGraphNode *a) { return log_impl(a, FATAN); }
/** Takes the elementwise square root of a */
FGraphNode *fsqrt_g(FGraphNode *a) { return log_impl(a, FSQRT); }
FGraphNode *fexp(FGraphNode *a) { return log_impl(a, FEXP); }
/** Negates the elements of the tensor */
FGraphNode *fneg(FGraphNode *a) {
	FOperation op;
	op.broadcasting_mode = 0;
	op.additional_data = nullptr;
	op.op_type = FNEG;
	op.dimensions = a->operation.dimensions;
	op.shape = safe_mal<size_t>(op.dimensions);
	if (!op.shape)
		return nullptr;
	memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
	op.data_type = a->operation.data_type;
	return addNode(op, {a});
}
FGraphNode *fsign(FGraphNode *a) {
	FOperation op;
	op.broadcasting_mode = 0;
	op.additional_data = nullptr;
	op.op_type = FSIGN;
	op.dimensions = a->operation.dimensions;
	op.shape = safe_mal<size_t>(op.dimensions);
	if (!op.shape)
		return nullptr;
	memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
	op.data_type = F_INT32;
	FGraphNode *g = addNode(op, {a});
	return g;
}
FGraphNode *feven(FGraphNode *a) {
	if (a->operation.data_type != F_INT32 &&
		a->operation.data_type != F_INT64) {
		last_error = WRONG_TYPE;
		flogging(F_ERROR,
				 "Can't compute if tensor is even for floating point tensor!");
		return nullptr; // for c compatibility
	}
	FOperation op;
	op.broadcasting_mode = 0;
	op.additional_data = nullptr;
	op.op_type = FEVEN;
	op.dimensions = a->operation.dimensions;
	op.shape = safe_mal<size_t>(op.dimensions);
	if (!op.shape)
		return nullptr;
	memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
	op.data_type = F_INT32;
	FGraphNode *g = addNode(op, {a});
	return g;
}
FGraphNode *fflatten(FGraphNode *a) {
	FOperation op;
	op.broadcasting_mode = 0;
	op.additional_data = nullptr;
	op.op_type = FLATTEN;
	op.dimensions = 1;
	op.shape = safe_mal<size_t>(1);
	if (!op.shape)
		return nullptr;
	const FOperation prev_op = a->operation;
	size_t total_size = 1;
	for (int i = 0; i < prev_op.dimensions; i++)
		total_size *= prev_op.shape[i];
	op.shape[0] = total_size;
	op.data_type = prev_op.data_type;
	return addNode(op, {a});
}
FGraphNode *fflatten_dimension(FGraphNode *a, const int dimension) {
	if (dimension == 0) {
		last_error = ILLEGAL_DIMENSION;
		flogging(F_ERROR,
				 "Flattening the first dimension of a tensor is not possible!");
		return nullptr; // for c compatibility
	}

	const FOperation prev_op = a->operation;
	size_t new_prevdim_size =
		prev_op.shape[dimension - 1] * prev_op.shape[dimension];
	FOperation op;
	op.broadcasting_mode = 0;
	op.op_type = FLATTEN;
	op.dimensions = prev_op.dimensions - 1;
	op.shape = safe_mal<size_t>(prev_op.dimensions - 1);
	if (!op.shape)
		return nullptr;
	// copy into shape
	memcpy(op.shape, prev_op.shape, sizeof(size_t) * dimension);
	memcpy(op.shape + dimension, prev_op.shape + (dimension + 1),
		   sizeof(size_t) * (prev_op.dimensions - dimension - 1));
	op.shape[dimension - 1] = new_prevdim_size;

	op.additional_data = nullptr;
	op.data_type = prev_op.data_type;
	return addNode(op, {a});
}

FGraphNode *fmatmul(FGraphNode *x, FGraphNode *y) {
	const FOperation ao = x->operation;
	const FOperation bo = y->operation;
	if (ao.dimensions < 2 || bo.dimensions < 2) {
		last_error = ILLEGAL_DIMENSIONALITY;
		flogging(F_ERROR, "Dimensions of operands of matrix multiplications "
						  "must be at least 2!");
		return nullptr;
	}
	const size_t l = ao.shape[ao.dimensions - 2];
	const size_t m = ao.shape[ao.dimensions - 1];
	// matmul(l x m, m x n)
	// = reduce_sum(mul(a, transpose(repeat(b, l)), {_, -1, -2, -3}), -2)
	const size_t mb = bo.shape[bo.dimensions - 2];
	const size_t n = bo.shape[bo.dimensions - 1];
	if (m != mb) {
		last_error = INCOMPATIBLE_SHAPES;
		flogging(F_ERROR, "Incompatible Shapes for matrix multiplications: " +
							  vector_string(std::vector<size_t>(
								  ao.shape, ao.shape + ao.dimensions)) +
							  " and " +
							  vector_string(std::vector<size_t>(
								  bo.shape, bo.shape + bo.dimensions)));
		return nullptr; // for c compatibility
	}
	FGraphNode* total = fmul(fexpand(x, ao.dimensions, n), fexpand(y, bo.dimensions - 2, l));
	return freduce_sum(total, total->operation.dimensions - 2);	
}
FGraphNode *freshape(FGraphNode *a, const size_t *newshape,
					 const int dimensions) {
	size_t total_size_node = 1;
	for (int i = 0; i < a->operation.dimensions; i++)
		total_size_node *= a->operation.shape[i];
	size_t total_size_new = 1;
	for (int i = 0; i < dimensions; i++)
		total_size_new *= newshape[i];
	if (total_size_node != total_size_new) {
		last_error = INCOMPATIBLE_SHAPES;
		flogging(F_ERROR, "To reshape a node the product of its new shape must "
						  "match the product of its old!");
		return nullptr; // for c compatibility
	}
	FGraphNode *node = new FGraphNode();
	configureGradientInformation(node, {a});
	node->result_data = nullptr;
	node->operation.shape = safe_mal<size_t>(dimensions);
	if (!node->operation.shape)
		return nullptr;
	std::memcpy(node->operation.shape, newshape, dimensions * sizeof(size_t));
	node->operation.data_type = a->operation.data_type;
	node->operation.op_type = FRESHAPE;
	node->operation.dimensions = dimensions;
	node->num_predecessor = 1;
	node->predecessors = safe_mal<FGraphNode *>(1);
	if (!node->predecessors)
		return nullptr;
	node->predecessors[0] = a;
	node->reference_counter = 0;
	if (a->reference_counter++ > 2)
		fExecuteGraph(a);
	return node;
}
FGraphNode *fconvert(FGraphNode *a, FType newtype) {
	FGraphNode *foo = new FGraphNode();
	configureGradientInformation(foo, {a});
	foo->reference_counter = 0;
	foo->num_predecessor = 1;
	foo->result_data = nullptr;
	foo->predecessors = safe_mal<FGraphNode *>(1);
	if (!foo->predecessors)
		return nullptr;
	foo->predecessors[0] = a;
	if (a->reference_counter++ > 2)
		fExecuteGraph(a);
	foo->operation.data_type = newtype;
	foo->operation.dimensions = a->operation.dimensions;
	foo->operation.shape = safe_mal<size_t>(a->operation.dimensions);
	if (!foo->operation.shape)
		return nullptr;
	memcpy(foo->operation.shape, a->operation.shape,
		   sizeof(size_t) * a->operation.dimensions);
	foo->operation.op_type = FCONVERSION;
	foo->operation.additional_data = nullptr;
	return foo;
}

static inline FGraphNode *reduce_operation(FGraphNode *a, const int dimension,
										   FOperationType type) {
	size_t total = 1;
	for (int i = 0; i < a->operation.dimensions; i++)
		if (i != dimension)
			total *= a->operation.shape[i];
	if (total <= 128 ||
		a->reference_counter > 1) { // small reduction size will be slow on gpu
		a = fExecuteGraph(a);
	} else if (!a->result_data) {
		// we dont want interleaved reduction since that is slow
		std::list<FGraphNode *> todo;
		todo.push_back(a);
		while (!todo.empty()) {
			FGraphNode *curr = todo.front();
			todo.pop_front();
			bool terminate = false;
			if (curr->result_data)
				continue;
			switch (curr->operation.op_type) {
			case FCONVOLVE:
			case FMATMUL:
			case FGRADIENT_CONVOLVE1:
			case FREDUCE_MAX:
			case FREDUCE_MIN:
			case FREDUCE_MUL:
			case FREDUCE_SUM:
				a = fExecuteGraph(a);
				terminate = true;
			default:
				break;
			}
			if (terminate)
				break;
			for (int i = 0; i < curr->num_predecessor; i++)
				todo.push_back(curr->predecessors[i]);
		}
	}
	FGraphNode *foo = new FGraphNode();
	configureGradientInformation(foo, {a});
	foo->reference_counter = 0;
	foo->num_predecessor = 1;
	foo->result_data = nullptr;
	foo->predecessors = safe_mal<FGraphNode *>(1);
	if (!foo->predecessors)
		return nullptr;
	foo->predecessors[0] = a;
	a->reference_counter++;
	FOperation op;
	const FOperation other = a->operation;
	op.broadcasting_mode = 0;
	op.data_type = other.data_type;
	op.op_type = type;
	if (other.dimensions > 1) {
		op.dimensions = other.dimensions - 1;
		op.shape = safe_mal<size_t>(op.dimensions);
		if (!op.shape)
			return nullptr;
		memcpy(op.shape, other.shape, sizeof(size_t) * dimension);
		memcpy(op.shape + dimension, other.shape + (dimension + 1),
			   sizeof(size_t) * (other.dimensions - dimension - 1));
	} else {
		op.dimensions = 1;
		op.shape = safe_mal<size_t>(1);
		if (!op.shape)
			return nullptr;
		op.shape[0] = 1;
	}
	op.additional_data = safe_mal<int>(1);
	if (!op.additional_data)
		return nullptr;
	((int *)op.additional_data)[0] = dimension;
	foo->operation = op;
	return foo;
}
// freduce_sum([[1,2,3], [4,5,6]], 0) = [5,7,9],
// freduce_sum([[1,2,3], [4,5,6]], 1) = [6,15]
FGraphNode *freduce_sum(FGraphNode *a, const int dimension) {
	return reduce_operation(a, dimension, FREDUCE_SUM);
}
FGraphNode *freduce_mul(FGraphNode *a, const int dimension) {
	return reduce_operation(a, dimension, FREDUCE_MUL);
}
FGraphNode *freduce_min(FGraphNode *a, const int dimension) {
	return reduce_operation(a, dimension, FREDUCE_MIN);
}
FGraphNode *freduce_max(FGraphNode *a, const int dimension) {
	return reduce_operation(a, dimension, FREDUCE_MAX);
}

FGraphNode *fslice_step(FGraphNode *a, const long *start, const long *end,
						const long *step) {
	// construct nodes
	FGraphNode *foo = new FGraphNode();
	configureGradientInformation(foo, {a});
	foo->num_predecessor = 1;
	foo->result_data = nullptr;
	foo->predecessors = safe_mal<FGraphNode *>(1);
	if (!foo->predecessors)
		return nullptr;
	foo->predecessors[0] = a;
	foo->reference_counter = 0;
	if (a->reference_counter++ > 2)
		fExecuteGraph(a);
	FOperation op;
	op.broadcasting_mode = 0;
	op.op_type = FSLICE;
	op.data_type = a->operation.data_type;
	op.dimensions = a->operation.dimensions;
	op.shape = safe_mal<size_t>(op.dimensions);
	if (!op.shape)
		return nullptr;
	FSlice *slice = new FSlice();
	op.additional_data = (void *)slice;
	slice->step = safe_mal<long>(op.dimensions);
	if (!slice->step)
		return nullptr;
	slice->start = safe_mal<long>(op.dimensions);
	if (!slice->start)
		return nullptr;
	slice->end = safe_mal<long>(op.dimensions);
	if (!slice->end)
		return nullptr;
	for (size_t i = 0; i < op.dimensions; i++) {
		if (step[i] == 0) {
			last_error = INVALID_SELECT;
			flogging(F_ERROR, "Step may not be 0 for slicing!");
			return nullptr; // for c compatibility
		}
		slice->start[i] =
			(start[i] < 0) ? (long)a->operation.shape[i] + start[i] : start[i];
		slice->end[i] =
			(end[i] < 0) ? (long)a->operation.shape[i] + end[i] : end[i];
		slice->step[i] = step[i];
		op.shape[i] = ABS(slice->end[i] - slice->start[i]);
		long step_abs = ABS(step[i]);
		// start element is always present
		if (op.shape[i] % step_abs == 0)
			op.shape[i] = op.shape[i] / step_abs;
		else
			op.shape[i] = op.shape[i] / step_abs + 1;
		if (op.shape[i] > a->operation.shape[i]) {
			last_error = INVALID_SELECT;
			flogging(F_ERROR, "Invalid slice: dimension " + std::to_string(i) +
								  " larger then target tensor! (" +
								  std::to_string(op.shape[i]) + " > " +
								  std::to_string(a->operation.shape[i]) + ")");
			return nullptr; // for c compatibility
		}
		if ((step[i] < 0 && (slice->end[i] > slice->start[i])) ||
			(step[i] > 0 && (slice->end[i] < slice->start[i]))) {
			last_error = INVALID_SELECT;
			flogging(F_ERROR,
					 "invalid slice: combination of step sign, start and end "
					 "in dimension " +
						 std::to_string(i) +
						 " will yield empty tensor! start: " +
						 std::to_string(slice->start[i]) +
						 ", end: " + std::to_string(slice->end[i]) +
						 ", step: " + std::to_string(slice->step[i]));
			return nullptr; // for c compatibility
		}
	}
	foo->operation = op;
	return foo;
}
FGraphNode *fslice(FGraphNode *a, const long *start, const long *end) {
	std::vector<long> step(a->operation.dimensions, 1);
	FGraphNode *foo = fslice_step(a, start, end, &step[0]);
	return foo;
}
FGraphNode *fabs_g(FGraphNode *a) {
	FOperation op;
	op.broadcasting_mode = 0;
	op.op_type = FABS;
	op.additional_data = nullptr;
	initShape_keep(op, &a->operation, nullptr);
	op.data_type = a->operation.data_type;
	return addNode(op, {a});
}
FGraphNode *frepeat(FGraphNode *a, int *repetitions) {
	FOperation op;
	op.broadcasting_mode = 0;
	op.op_type = FREPEAT;
	op.data_type = a->operation.data_type;
	op.dimensions = a->operation.dimensions;
	op.shape = safe_mal<size_t>(op.dimensions);
	if (!op.shape)
		return nullptr;
	for (int dim = 0; dim < op.dimensions; dim++) {
		op.shape[dim] = a->operation.shape[dim] * (repetitions[dim] + 1);
	}
	op.data_type = a->operation.data_type;
	op.additional_data = nullptr;
	return addNode(op, {a});
}
FGraphNode *ftranspose(FGraphNode *a, int *transpositions) {
	FOperation op;
	op.broadcasting_mode = 0;
	op.op_type = FTRANSPOSE;
	op.data_type = a->operation.data_type;
	op.dimensions = a->operation.dimensions;
	op.shape = safe_mal<size_t>(op.dimensions);
	if (!op.shape)
		return nullptr;
	for (int i = 0; i < op.dimensions; i++) {
		op.shape[i] = a->operation.shape[transpositions[i]];
		// check that transpositions is reflexive
		if (transpositions[transpositions[i]] != i)
			flogging(
				F_ERROR,
				"Transpositions Array must be reflexive i.e for an dimension i "
				"let j "
				"be transpositions[i]. Then i = transpositions[j] must hold.");
	}
	op.additional_data = safe_mal<int>(op.dimensions);
	if (!op.additional_data)
		return nullptr;
	memcpy(op.additional_data, transpositions,
		   sizeof(int) * a->operation.dimensions);
	op.data_type = a->operation.data_type;
	return addNode(op, {a});
}
FGraphNode *fless_g(FGraphNode *a, FGraphNode *b) {
	FOperation op;
	op.op_type = FLESS;
	op.additional_data = nullptr;
	initShape_keep(op, &a->operation, &b->operation);
	op.data_type = F_INT32;
	FGraphNode *g = addNode(op, {a, b});
	return g;
}
FGraphNode *fgreater_g(FGraphNode *a, FGraphNode *b) {
	FOperation op;
	op.op_type = FGREATER;
	op.additional_data = nullptr;
	initShape_keep(op, &a->operation, &b->operation);
	op.data_type = F_INT32;
	FGraphNode *g = addNode(op, {a, b});
	return g;
}
FGraphNode *fequal_g(FGraphNode *a, FGraphNode *b) {
	FOperation op;
	op.op_type = FEQUAL;
	op.additional_data = nullptr;
	initShape_keep(op, &a->operation, &b->operation);
	op.data_type = F_INT32;
	FGraphNode *g = addNode(op, {a, b});
	return g;
}
template <typename T> static inline FGraphNode *less(FGraphNode *a, const T b) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FLESS;
	initShape_keep(op, &a->operation, nullptr);
	FGraphNode *g = addNodeWithConst(op, a, b);
	g->operation.data_type = F_INT32;
	return g;
}
FGraphNode *fless_ci(FGraphNode *a, const int b) { return less(a, b); }
FGraphNode *fless_cl(FGraphNode *a, const long b) { return less(a, b); }
FGraphNode *fless_cf(FGraphNode *a, const float b) { return less(a, b); }
FGraphNode *fless_cd(FGraphNode *a, const double b) { return less(a, b); }

template <typename T>
static inline FGraphNode *greater(FGraphNode *a, const T b) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FGREATER;
	initShape_keep(op, &a->operation, nullptr);
	op.data_type = F_INT32;
	FGraphNode *g = addNodeWithConst(op, a, b);
	return g;
}
FGraphNode *fgreater_ci(FGraphNode *a, const int b) { return greater(a, b); }
FGraphNode *fgreater_cl(FGraphNode *a, const long b) { return greater(a, b); }
FGraphNode *fgreater_cf(FGraphNode *a, const float b) { return greater(a, b); }
FGraphNode *fgreater_cd(FGraphNode *a, const double b) { return greater(a, b); }

template <typename T>
static inline FGraphNode *equal(FGraphNode *a, const T b) {
	FOperation op;
	op.additional_data = nullptr;
	op.op_type = FEQUAL;
	initShape_keep(op, &a->operation, nullptr);
	op.data_type = F_INT32;
	FGraphNode *g = addNodeWithConst(op, a, b);
	return g;
}
FGraphNode *fequal_ci(FGraphNode *a, const int b) { return equal(a, b); }
FGraphNode *fequal_cl(FGraphNode *a, const long b) { return equal(a, b); }
FGraphNode *fequal_cf(FGraphNode *a, const float b) { return equal(a, b); }
FGraphNode *fequal_cd(FGraphNode *a, const double b) { return equal(a, b); }
FGraphNode *fextend_step(FGraphNode *a, const size_t *new_shape,
						 const size_t *insert_at, const long *step_size) {
	// construct nodes
	FGraphNode *foo = new FGraphNode();
	configureGradientInformation(foo, {a});
	foo->num_predecessor = 1;
	foo->result_data = nullptr;
	foo->predecessors = safe_mal<FGraphNode *>(1);
	if (!foo->predecessors)
		return nullptr;
	foo->predecessors[0] = a;
	foo->reference_counter = 0;
	if (a->reference_counter++ > 2)
		fExecuteGraph(a);
	// construct operation
	const int dimensions = a->operation.dimensions;
	FOperation op;
	op.broadcasting_mode = 0;
	op.op_type = FEXTEND;
	op.data_type = a->operation.data_type;
	op.dimensions = dimensions;
	op.shape = safe_mal<size_t>(dimensions);
	if (!op.shape)
		return nullptr;
	memcpy(op.shape, new_shape, dimensions * sizeof(size_t));
	// set the parallel score
	op.additional_data = new FExtend();
	foo->operation = op;
	FExtend &extend = *(FExtend *)op.additional_data;
	extend.start = safe_mal<size_t>(dimensions);
	if (!extend.start)
		return nullptr;
	extend.step = safe_mal<long>(dimensions);
	if (!extend.step)
		return nullptr;
	memcpy(extend.start, insert_at, dimensions * sizeof(size_t));
	memcpy(extend.step, step_size, dimensions * sizeof(long));
	return foo;
}
FGraphNode *fextend(FGraphNode *a, const size_t *new_shape,
					const size_t *insert_at) {
	const int dimensions = a->operation.dimensions;
	std::vector<long> steps(dimensions, 1);
	return fextend_step(a, new_shape, insert_at, steps.data());
}
FGraphNode *fconcat(FGraphNode *a, FGraphNode *b, const unsigned int axis) {
	FOperation op;
	op.broadcasting_mode = 0;
	op.op_type = FCONCAT;
	op.dimensions = a->operation.dimensions;
	op.shape = safe_mal<size_t>(a->operation.dimensions);
	if (!op.shape)
		return nullptr;
	std::memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
	op.shape[axis] = a->operation.shape[axis] + b->operation.shape[axis];
	for (int i = 0; i < op.dimensions; i++)
		if (i != axis && a->operation.shape[i] != b->operation.shape[i]) {
			last_error = INCOMPATIBLE_SHAPES;
			flogging(
				F_ERROR,
				"Concatenations of two nodes excpects both to have the same "
				"size along every dimension except the concatenation one!");
			return nullptr; // for c compatibility
		}
	op.data_type = a->operation.data_type;
	op.additional_data = safe_mal<unsigned int>(1);
	if (!op.additional_data)
		return nullptr;
	((unsigned int *)op.additional_data)[0] = axis;
	return addNode(op, {a, b});
}
FGraphNode *fexpand(FGraphNode *a, const unsigned int ax,
					const unsigned int ax_size) {
	unsigned int n = a->operation.dimensions;
	std::vector<size_t> new_shape(n + 1);
	if (ax > 0)
		std::memcpy(new_shape.data(), a->operation.shape,
					sizeof(size_t) * std::min(n, ax));
	new_shape[ax] = 1;
	if (ax < n)
		std::memcpy(new_shape.data() + ax + 1, a->operation.shape + ax,
					sizeof(size_t) * (n - ax));
	if (ax_size == 0)
		return freshape(a, new_shape.data(), n + 1);
	std::vector<int> repet(n + 1, 0);
	repet[ax] = ax_size - 1;
	FGraphNode *res = freshape(a, new_shape.data(), n + 1);
	return ax_size == 1 ? res : frepeat(res, repet.data());
}
/** Calculates the shape for a sliding window operation, that accumulates all
 * elements in a window. target.shape should already be allocated */
static void calculateShapeAggregatingWindows(FOperation &target,
											 const FOperation &orig,
											 const size_t *size,
											 const unsigned int *steps) {
	for (int i = 0; i < orig.dimensions - 1; i++) {
		const size_t kernel_shape = size[i];
		size_t window_size = orig.shape[i] - kernel_shape + 1;
		window_size = window_size % steps[i] == 0 ? window_size / steps[i]
												  : window_size / steps[i] + 1;
		target.shape[i] = window_size;
	}
}
FGraphNode *fconvolve(FGraphNode *a, FGraphNode *kernel,
					  const unsigned int *steps) {
	const FOperation ao = a->operation;
	const FOperation bo = kernel->operation;
	if (!a->result_data && ao.op_type != FSTORE) {
		fExecuteGraph(a);
	}
	if (!kernel->result_data && bo.op_type != FSTORE) {
		fExecuteGraph(kernel);
	}
	if (ao.dimensions != bo.dimensions && ao.dimensions + 1 != bo.dimensions) {
		last_error = ILLEGAL_DIMENSIONALITY;
		flogging(F_ERROR,
				 "For a convolution the original Tensor and the filter "
				 "kernel(s) have to have to same number of dimensions!");
		return nullptr; // for c compatibility
	}
	bool multiple_filters = ao.dimensions + 1 == bo.dimensions;
	if (ao.shape[ao.dimensions - 1] != bo.shape[bo.dimensions - 1]) {
		last_error = INCOMPATIBLE_SHAPES;
		flogging(F_ERROR,
				 "For a convolution the size of the last dimension of the "
				 "Tensor must match that of the kernel! " +
					 std::to_string(ao.shape[ao.dimensions - 1]) + " vs. " +
					 std::to_string(bo.shape[bo.dimensions - 1]));
		return nullptr; // for c compatibility
	}
	FOperation op;
	op.broadcasting_mode = 0;
	op.dimensions = multiple_filters ? ao.dimensions : ao.dimensions - 1;
	op.shape = safe_mal<size_t>(op.dimensions);
	if (!op.shape)
		return nullptr;
	calculateShapeAggregatingWindows(
		op, ao, multiple_filters ? bo.shape + 1 : bo.shape, steps);
	if (multiple_filters)
		op.shape[ao.dimensions - 1] = bo.shape[0];
	op.data_type = higher_type(ao.data_type, bo.data_type);
	op.op_type = FCONVOLVE;
	op.additional_data = safe_mal<unsigned int>(op.dimensions);
	if (!op.additional_data)
		return nullptr;
	memcpy(op.additional_data, steps, op.dimensions * sizeof(unsigned int));
	return addNode(op, {a, kernel});
}
FGraphNode *frandom(const size_t *shape, const int dimensions) {
	FGraphNode *node = new FGraphNode();
	FOperation op;
	op.broadcasting_mode = 0;
	op.op_type = FGEN_RANDOM;
	op.dimensions = dimensions;
	op.shape = safe_mal<size_t>(dimensions);
	if (!op.shape)
		return nullptr;
	memcpy(op.shape, shape, dimensions * sizeof(size_t));
	op.data_type = F_FLOAT64;
	// Store current time in additional data
	std::chrono::duration<double, std::nano> tm =
		std::chrono::high_resolution_clock::now().time_since_epoch();
	double t = ((unsigned long)tm.count() % 1000000) / 100.0;
	op.additional_data = safe_mal<double>(1);
	if (!op.additional_data)
		return nullptr;
	((double *)op.additional_data)[0] = t;
	node->operation = op;
	node->result_data = nullptr;
	node->predecessors = nullptr;
	node->num_predecessor = 0;
	node->gradient_data = nullptr;
	node->reference_counter = 0;
	return node;
}
FGraphNode *fdropout(FGraphNode *g, const double p) {
	FOperation op;
	op.broadcasting_mode = 0;
	op.op_type = FDROPOUT;
	op.dimensions = g->operation.dimensions;
	op.shape = safe_mal<size_t>(g->operation.dimensions);
	if (!op.shape)
		return nullptr;
	memcpy(op.shape, g->operation.shape,
		   g->operation.dimensions * sizeof(size_t));
	op.data_type = g->operation.data_type;
	// Store current time in additional data
	std::chrono::duration<double, std::nano> tm =
		std::chrono::high_resolution_clock::now().time_since_epoch();
	double t = ((unsigned long)tm.count() % 1000000) / 100.0;
	op.additional_data = safe_mal<double>(2);
	if (!op.additional_data)
		return nullptr;
	((double *)op.additional_data)[0] = t;
	((double *)op.additional_data)[1] = p;
	return addNode(op, {g});
}
FGraphNode *findex(FGraphNode *a, FGraphNode *indices) {
	if (indices->operation.dimensions > a->operation.dimensions) {
		last_error = ILLEGAL_DIMENSIONALITY;
		flogging(
			F_ERROR,
			"Invalid index Tensor dimensionality! Larger than indexed Tensor!");
		return nullptr; // for c compatibility
	}
	if (indices->operation.data_type != F_INT32 &&
		indices->operation.data_type != F_INT64) {
		last_error = WRONG_TYPE;
		flogging(F_ERROR, "Only integer tensors may be used as indices!");
		return nullptr; // for c compatibility
	}
	for (int d = 0; d < indices->operation.dimensions - 1; d++)
		if (a->operation.shape[d] != indices->operation.shape[d]) {
			last_error = INCOMPATIBLE_SHAPES;
			flogging(
				F_ERROR,
				"Invalid indices shape! Except for last dimension shape of "
				"indices Tensor has to be a prefix of the indexed Tensor!");
			return nullptr; // for c compatibility
		}

	FOperation op;
	op.broadcasting_mode = 0;
	op.op_type = FINDEX;
	op.dimensions = a->operation.dimensions;
	op.shape = safe_mal<size_t>(op.dimensions);
	if (!op.shape)
		return nullptr;
	memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
	op.shape[indices->operation.dimensions - 1] =
		indices->operation.shape[indices->operation.dimensions - 1];
	op.data_type = a->operation.data_type;
	op.additional_data = nullptr;
	return addNode(op, {a, indices});
}
FGraphNode *findex_set(FGraphNode *a, FGraphNode *b, FGraphNode *indices) {
	if (!indices->result_data && indices->operation.op_type != FSTORE)
		indices = fExecuteGraph(indices);
	if (!b->result_data && b->operation.op_type != FSTORE)
		b = fExecuteGraph(b);
	if (indices->operation.dimensions > b->operation.dimensions) {
		last_error = ILLEGAL_DIMENSIONALITY;
		flogging(
			F_ERROR,
			"Invalid index Tensor dimensionality! Larger than indexed Tensor!");
		return nullptr; // for c compatibility
	}
	if (indices->operation.data_type != F_INT32 &&
		indices->operation.data_type != F_INT64) {
		last_error = WRONG_TYPE;
		flogging(F_ERROR, "Only integer tensors may be used as indices!");
		return nullptr; // for c compatibility
	}
	for (int d = 0; d < indices->operation.dimensions - 1; d++)
		if (b->operation.shape[d] != indices->operation.shape[d]) {
			last_error = INCOMPATIBLE_SHAPES;
			flogging(
				F_ERROR,
				"Invalid indices shape! Except for last dimension shape of "
				"indices Tensor has to be a prefix of the indexed Tensor!");
			return nullptr; // for c compatibility
		}

	FOperation op;
	op.broadcasting_mode = 0;
	op.op_type = FSET_INDEX;
	op.dimensions = a->operation.dimensions;
	op.shape = safe_mal<size_t>(op.dimensions);
	if (!op.shape)
		return nullptr;
	memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
	op.data_type = a->operation.data_type;
	op.additional_data = nullptr;
	return addNode(op, {a, b, indices});
}
FGraphNode *fsliding_window(FGraphNode *a, const size_t *size,
							const unsigned int *steps) {
	FOperation op;
	op.broadcasting_mode = 0;
	op.op_type = FSLIDING_WINDOW;
	op.dimensions = a->operation.dimensions + 1;
	op.data_type = a->operation.data_type;
	op.shape = safe_mal<size_t>(op.dimensions);
	if (!op.shape)
		return nullptr;
	op.shape[0] = 1;
	for (int i = 0; i < a->operation.dimensions; i++) {
		op.shape[i + 1] = size[i];
		// we slide a window of size size[i] with step size steps[i] along that
		// dimension
		size_t window_size = a->operation.shape[i] - size[i] + 1;
		window_size = window_size % steps[i] == 0 ? window_size / steps[i]
												  : window_size / steps[i] + 1;
		op.shape[0] *= window_size;
	}
	FSlidingWindow *slidewin = new FSlidingWindow();
	slidewin->size = safe_mal<size_t>(a->operation.dimensions);
	if (!slidewin->size)
		return nullptr;
	slidewin->step = safe_mal<unsigned int>(a->operation.dimensions);
	if (!slidewin->step)
		return nullptr;
	memcpy(slidewin->size, size, a->operation.dimensions * sizeof(size_t));
	memcpy(slidewin->step, steps,
		   a->operation.dimensions * sizeof(unsigned int));
	op.additional_data = (void *)(slidewin);
	return addNode(op, {a});
}
FGraphNode *funslide_window(FGraphNode *a, const size_t *shape,
							const unsigned int *steps) {
	if (!a->result_data && a->operation.op_type != FSTORE)
		fExecuteGraph(a);
	FOperation op;
	op.broadcasting_mode = 0;
	op.op_type = FUNSLIDE_WINDOW;
	op.dimensions = a->operation.dimensions - 1;
	op.data_type = a->operation.data_type;
	op.shape = safe_mal<size_t>(op.dimensions);
	if (!op.shape)
		return nullptr;
	size_t no_windows = 1;
	for (int i = 0; i < a->operation.dimensions - 1; i++) {
		size_t window_size = shape[i] - a->operation.shape[i + 1] + 1;
		window_size = window_size % steps[i] == 0 ? window_size / steps[i]
												  : window_size / steps[i] + 1;
		no_windows *= window_size;
		op.shape[i] = shape[i];
	}
	if (no_windows != a->operation.shape[0]) {
		last_error = INCOMPATIBLE_SHAPES;
		flogging(F_ERROR,
				 "Number of windows is not consistend with provided shape "
				 "and steps for unslide! Provided parameters yield " +
					 std::to_string(no_windows) +
					 " windows, while the provided Tensor has " +
					 std::to_string(a->operation.shape[0]));
		return nullptr; // for c compatibility
	}
	unsigned int *csteps = safe_mal<unsigned int>(op.dimensions);
	if (!csteps)
		return nullptr;
	memcpy(csteps, steps, op.dimensions * sizeof(unsigned int));
	op.additional_data = csteps;
	return addNode(op, {a});
}
FGraphNode *fpermutate(FGraphNode *a, unsigned int ax) {
	size_t total_size;
	const long *perms =
		generate_permutation(a->operation.shape, ax, &total_size);
	if (!perms)
		return nullptr;
	FGraphNode *ind =
		fCreateGraph(perms, total_size, F_INT64, a->operation.shape, ax + 1);
	if (!ind)
		return nullptr;
	return findex(a, ind);
}
FGraphNode *fpooling_sum(FGraphNode *a, const size_t *window_size,
						 const unsigned int *step_size) {
	FOperation op;
	op.dimensions = a->operation.dimensions - 1;
	op.shape = safe_mal<size_t>(op.dimensions);
	if (!op.shape)
		return nullptr;
	calculateShapeAggregatingWindows(op, a->operation, window_size, step_size);
	op.op_type = FPOOLING_SUM;
	op.data_type = a->operation.data_type;
	FSlidingWindow *window = new FSlidingWindow();
	window->size = safe_mal<size_t>(op.dimensions);
	window->step = safe_mal<unsigned int>(op.dimensions);
	memcpy(window->size, window_size, sizeof(size_t) * op.dimensions);
	memcpy(window->step, step_size, sizeof(unsigned int) * op.dimensions);
	op.additional_data = window;
	op.broadcasting_mode = 0;
	return addNode(op, {a});
}
FGraphNode *fpooling_max(FGraphNode *a, const size_t *window_size,
						 const unsigned int *step_size) {
	FOperation op;
	op.dimensions = a->operation.dimensions - 1;
	op.shape = safe_mal<size_t>(op.dimensions);
	if (!op.shape)
		return nullptr;
	calculateShapeAggregatingWindows(op, a->operation, window_size, step_size);
	op.op_type = FPOOLING_MAX;
	op.data_type = a->operation.data_type;
	FSlidingWindow *window = new FSlidingWindow();
	window->size = safe_mal<size_t>(op.dimensions);
	window->step = safe_mal<unsigned int>(op.dimensions);
	memcpy(window->size, window_size, sizeof(size_t) * op.dimensions);
	memcpy(window->step, step_size, sizeof(unsigned int) * op.dimensions);
	op.additional_data = window;
	op.broadcasting_mode = 0;
	return addNode(op, {a});
}
