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
#include "../../flint.h"
#include "../operations/implementation.hpp"
#include "../utils.hpp"
#include "codegen.hpp"
#include <list>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

/** Largest number of elements of any tensor that occurs in the kernel, i.e.
 * the bound for every index calculation in it */
static size_t maximumTensorSize(FGraphNode *node) {
	using namespace std;
	size_t max_size = 0;
	unordered_set<const FGraphNode *> visited;
	list<FGraphNode *> todo = {node};
	while (!todo.empty()) {
		FGraphNode *curr = todo.front();
		todo.pop_front();
		if (!visited.insert(curr).second)
			continue;
		size_t size = 1;
		for (int i = 0; i < curr->operation.dimensions; i++)
			size *= curr->operation.shape[i];
		max_size = max(max_size, size);
		// parameters are read directly, their predecessors are not part of
		// the kernel
		if (curr != node &&
			(curr->operation.op_type == FSTORE || curr->result_data ||
			 curr->operation.op_type == FGEN_CONSTANT))
			continue;
		for (int i = 0; i < curr->num_predecessor; i++)
			todo.push_back(curr->predecessors[i]);
	}
	return max_size;
}
/** True if a node is read from a buffer instead of being calculated */
static bool isKernelParameter(const FGraphNode *node) {
	return node->operation.op_type == FSTORE || node->result_data ||
		   node->operation.op_type == FGEN_CONSTANT;
}
/** True if the operation evaluates its predecessors with the same `index` it
 * was called with. Only those pass the global id down unchanged, everything
 * else remaps the index for its predecessors (reductions, slices, windows,
 * ...) so that the same node means different values in different places. */
static bool passesIndexOn(const FGraphNode *node) {
	switch (node->operation.op_type) {
	case FADD:
	case FSUB:
	case FMUL:
	case FDIV:
	case FPOW:
	case FMIN:
	case FMAX:
	case FLESS:
	case FEQUAL:
	case FGREATER: {
		// inverse broadcasting divides the index for the predecessors
		size_t iv1 = 1, iv2 = 1;
		calculate_divisor_for_inverse_broadcasting(node->predecessors[0], iv1,
												   node->predecessors[1], iv2);
		return iv1 == 1 && iv2 == 1;
	}
	case FNEG:
	case FLOG:
	case FLOG2:
	case FLOG10:
	case FSIGN:
	case FEVEN:
	case FSIN:
	case FCOS:
	case FTAN:
	case FASIN:
	case FACOS:
	case FATAN:
	case FSQRT:
	case FEXP:
	case FABS:
	case FCONVERSION:
	case FLATTEN:
	case FRESHAPE:
		return true;
	default:
		return false;
	}
}
/** Collects the nodes that are reachable from `node` without the index being
 * remapped, together with how many of those use them and how far they are away
 * from the root at most. Nodes with more than one user are written into the
 * kernel once instead of once per path to them. */
static void collectReusable(FGraphNode *root,
							std::unordered_map<FGraphNode *, int> &users,
							std::unordered_map<FGraphNode *, int> &distance) {
	using namespace std;
	// count the users of each node, expanding every node only once
	unordered_set<FGraphNode *> expanded;
	list<FGraphNode *> todo = {root};
	distance[root] = 0;
	while (!todo.empty()) {
		FGraphNode *curr = todo.front();
		todo.pop_front();
		if (!expanded.insert(curr).second)
			continue;
		if (isKernelParameter(curr) || !passesIndexOn(curr))
			continue;
		for (int i = 0; i < curr->num_predecessor; i++) {
			FGraphNode *pred = curr->predecessors[i];
			if (isKernelParameter(pred))
				continue;
			users[pred]++;
			todo.push_back(pred);
		}
	}
	// longest distance from the root, so that a node is always further away
	// than everything using it
	todo = {root};
	while (!todo.empty()) {
		FGraphNode *curr = todo.front();
		todo.pop_front();
		if (isKernelParameter(curr) || !passesIndexOn(curr))
			continue;
		for (int i = 0; i < curr->num_predecessor; i++) {
			FGraphNode *pred = curr->predecessors[i];
			if (isKernelParameter(pred))
				continue;
			if (distance[pred] < distance[curr] + 1) {
				distance[pred] = distance[curr] + 1;
				todo.push_back(pred);
			}
		}
	}
}
std::string
generateCode(FGraphNode *node,
			 std::list<std::pair<FGraphNode *, std::string>> &parameters,
			 std::vector<std::pair<FType, double>> &scalars) {
	using namespace std;
	OCLLazyCodegenState state;
	state.parameters = &parameters;
	state.code = {};
	// leave room for intermediate results of the index calculations
	state.index_type = maximumTensorSize(node) < (1l << 30) ? "int" : "long";
	const string &itype = state.index_type;
	// we use breadth first search to traverse to operation graph
	list<CodegenTask> &todo = state.todo;
	// some operations work on the parameters, allow them to keep track
	unordered_map<FGraphNode *, std::string> &assigned_params =
		state.assigned_params;
	unsigned int &variable_index = state.variable_index;
	Twine &code = state.code;
	// indexing logic (we save the old index in old_index$i to restore it)
	unsigned int &num_indices = state.num_indices;
	// nodes that more than one other node reads with the same index are
	// calculated once into `reused` instead of once per path to them
	unordered_map<FGraphNode *, int> users, distance;
	collectReusable(node, users, distance);
	unordered_map<FGraphNode *, string> reused;
	// the node currently generated as a reused definition, it may not alias
	// itself
	FGraphNode *defining = nullptr;
	todo.push_front({node, "v0", true});
	while (true) {
		if (todo.empty()) {
			// everything is generated, the definitions of the reused nodes
			// follow. The one closest to the root goes first so that it ends up
			// last in the code, after everything it reads.
			if (reused.empty())
				break;
			FGraphNode *next = nullptr;
			for (const auto &[gn, var] : reused)
				if (!next || distance[gn] < distance[next])
					next = gn;
			defining = next;
			todo.push_front({next, reused[next], true});
			reused.erase(next);
		}
		// take from queue
		const auto [node, name, same_index] = todo.front();
		todo.pop_front();
		state.index_defs = "";
		// used to insert code at a specific place
		if (!node) {
			code.prepend(name);
			continue;
		}
		// cash var
		string type = type_string(node->operation.data_type);
		bool push_pred = true;
		// a node that is used more than once is only calculated once, all
		// other uses refer to that variable. The definitions are generated
		// after the queue ran empty, so they end up before all of their uses.
		if (node != defining && same_index && users[node] > 1) {
			auto known = reused.find(node);
			if (known == reused.end())
				known = reused.insert({node, "v" + to_string(++variable_index)})
							.first;
			code.prepend("const " + type + " " + name + " = " + known->second +
						 ";\n");
			continue;
		}
		// write code
		const string opstr = string(fop_to_string[node->operation.op_type]);
		bool inverse_broadcasting =
			false; // adds index manipulation code for inverse broadcasting
		// a constant is a single value, passing it as an argument spares a
		// buffer allocation and upload per execution
		if (node->operation.op_type == FGEN_CONSTANT) {
			const FType dt = node->operation.data_type;
			double value = 0;
			switch (dt) {
			case F_INT32:
				value = ((int *)node->operation.additional_data)[0];
				break;
			case F_INT64:
				value = ((long *)node->operation.additional_data)[0];
				break;
			case F_FLOAT32:
				value = ((float *)node->operation.additional_data)[0];
				break;
			case F_FLOAT64:
				value = ((double *)node->operation.additional_data)[0];
				break;
			}
			code.prepend("const " + type + " " + name + " = " +
						 state.addScalar(dt, value) + ";\n");
			continue;
		}
		// need to be outside switch to include result_data
		if (node->operation.op_type == FSTORE || node->result_data) {
			push_pred = false;
			size_t num_entries =
				node->operation.op_type == FSTORE
					? ((FStore *)node->operation.additional_data)->num_entries
					: node->result_data->num_entries;
			if (assigned_params.find(node) == assigned_params.end()) {
				size_t pid = assigned_params.size();
				assigned_params.insert({node, "P" + to_string(pid)});
				parameters.push_back({node, "P" + to_string(pid)});
			}
			code.prepend("const " + type + " " + name + " = " +
						 assigned_params[node] + "[index%" +
						 to_string(num_entries) + "];\n");
		} else {
			const int flags =
				OperationImplementation::implementations[node->operation
															 .op_type]
					->generate_ocl_lazy(node, name, state);
			inverse_broadcasting =
				flags & OperationImplementation::OCL_LAZY_INVERSE_BROADCASTING;
			push_pred =
				(flags & OperationImplementation::OCL_LAZY_DONT_PUSH_PREDS) ==
				0;
		}
		if (inverse_broadcasting) {
			// manipulate for invserse broadcasting
			size_t iv1 = 1, iv2 = 1;
			calculate_divisor_for_inverse_broadcasting(
				node->predecessors[0], iv1, node->predecessors[1], iv2);
			if (iv1 != 1 || iv2 != 1) {
				push_pred = false;
				const string old_idx = "old_idx" + to_string(num_indices++);
				code.prepend("index = " + old_idx + ";\n");
				const int var1 = ++variable_index;
				const int var2 = ++variable_index;
				todo.push_front({nullptr, itype + " " + old_idx +
											  " = index;\nindex /= " +
											  to_string(iv2) + ";\n"});
				todo.push_front({node->predecessors[1], "v" + to_string(var2)});
				todo.push_front({nullptr, "index = " + old_idx +
											  ";\nindex /= " + to_string(iv1) +
											  ";\n"});
				todo.push_front({node->predecessors[0], "v" + to_string(var1)});
			}
		}
#ifdef FLINT_DEBUG
		code.prepend("// " + opstr + "\n");
#endif
		// insert our indexing logic into the queue after the children
		const string index_defs = state.index_defs;
		if (!index_defs.empty())
			todo.push_front({nullptr, index_defs});
		// push predecessors dfs
		if (push_pred) {
			const bool pred_same_index = same_index && passesIndexOn(node);
			for (int i = 0; i < node->num_predecessor; i++) {
				string parname = "v" + to_string(++variable_index);
				todo.push_front(
					{node->predecessors[i], parname, pred_same_index});
			}
		}
	}
	code.prepend(itype + " index = get_global_id(0);\n");
	scalars = std::move(state.scalars);
	return code;
}
