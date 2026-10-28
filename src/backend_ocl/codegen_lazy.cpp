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
#include "src/operations/indices.hpp"
#include <list>
#include <string>
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
	const LaunchRange range = launchRange(node);
	OCLLazyCodegenState state;
	state.parameters = &parameters;
	state.code = {};
	state.index_type = maximumTensorSize(node) < (1l << 31) ? "int" : "long";
	// the coordinates of the result, seeded from the launch ids exactly like
	// `launchRange` assigns them: the two innermost dimensions that are larger
	// than one get an id of their own, all remaining ones share the third
	Twine index_decl;
	state.index_map.expr_per_dim.resize(node->operation.dimensions);
	{
		int found = 0;
		size_t factor = 1;
		for (int i = node->operation.dimensions - 1; i >= 0; i--) {
			IndexExpr &coord = state.index_map.expr_per_dim[i];
			coord.a = 0;
			const size_t size = node->operation.shape[i];
			// a dimension of size one is always indexed with zero, it needs
			// neither a variable nor an id
			if (size == 1) {
				coord.derive_bound(state.index_atoms);
				continue;
			}
			std::string value;
			if (found < 2)
				value = "get_global_id(" + to_string(found) + ")";
			else {
				value = "(get_global_id(2)";
				if (factor > 1)
					value += " / " + to_string(factor);
				value += ") % " + to_string(size);
				factor *= size;
			}
			const int atom = state.add_atom("index_dim" + to_string(i), 0,
											(long)size - 1, 0);
			coord.exprs.push_back({1, atom});
			coord.derive_bound(state.index_atoms);
			index_decl.append(state.type_of(state.index_atoms[atom].bound) +
							  " " + state.index_atoms[atom].name + " = " +
							  value + ";\n");
			found++;
		}
	}
	const string &itype = state.index_type;
	// we use breadth first search to traverse to operation graph
	list<CodegenTask> &todo = state.todo;
	// some operations work on the parameters, allow them to keep track
	unordered_map<FGraphNode *, std::string> &assigned_params =
		state.assigned_params;
	unsigned int &variable_index = state.variable_index;
	Twine &code = state.code;
	// nodes that more than one other node reads with the same index are
	// calculated once into `reused` instead of once per path to them
	unordered_map<FGraphNode *, int> users, distance;
	collectReusable(node, users, distance);
	unordered_map<FGraphNode *, string> reused;
	// the node currently generated as a reused definition
	FGraphNode *defining = nullptr;
	// the coordinates of the result
	const IndexMap root_map = state.index_map;
	todo.push_front({node, "v0", true, state.index_map});
	while (true) {
		if (todo.empty()) {
			if (reused.empty())
				break;
			FGraphNode *next = nullptr;
			for (const auto &[gn, var] : reused)
				if (!next || distance[gn] < distance[next])
					next = gn;
			defining = next;
			// a reused node is only shared where the index is still the
			// global id
			state.index_map = root_map;
			IndexMap root_coords = root_map;
			IndexMap map = split_atoms(
				state.flatten(root_coords, node->operation.shape,
							  node->operation.dimensions),
				next->operation.shape, next->operation.dimensions, state);
			todo.push_front({next, reused[next], true, map});
			reused.erase(next);
		}
		// take from queue
		const auto [node, name, same_index, index_map, inserted_by,
					declares_indices, opens_loop] = todo.front();
		todo.pop_front();
		state.pred_index_maps.clear();
		state.pred_wrappers.clear();
		state.index_map = index_map;
		// code that is inserted at a specific place
		if (!node) {
			state.insert_text(name, index_map.current_depth, inserted_by,
							  declares_indices, opens_loop);
			continue;
		}
		// tells the index expressions this operation registers apart from
		// those of the operations below it
		state.generated_ops++;
		state.first_text = true;
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
			const IndexExpr flat =
				state.flatten(state.index_map, node->operation.shape,
							  node->operation.dimensions);
			string address = state.to_code(flat);
			if (flat.bound.first < 0 ||
				flat.bound.second >= (long)num_entries) {
				if (address.find(' ') != string::npos && address[0] != '(')
					address = "(" + address + ")";
				address += " % " + to_string(num_entries);
			}
			code.prepend("const " + type + " " + name + " = " +
						 assigned_params[node] + "[" + address + "];\n");
		} else {
			OperationImplementation *impl =
				OperationImplementation::implementations[node->operation
															 .op_type];
			const bool has_map = state.index_map.expr_per_dim.size() ==
								 node->operation.dimensions;
			// where are the parameters read? Only possible if the coordinates
			// of the node itself are known here
			if (has_map)
				impl->mutate_index(node, state);
			// mapping only some parameters would leave the others without an
			// index, the operation has to map either all of them or none
			if (!state.pred_index_maps.empty() &&
				state.pred_index_maps.size() != node->num_predecessor)
				flogging(F_ERROR,
						 "mutate_index of " +
							 string(fop_to_string[node->operation.op_type]) +
							 " mapped only some of its parameters.");
			if (!state.pred_wrappers.empty() &&
				state.pred_wrappers.size() != node->num_predecessor)
				flogging(F_ERROR,
						 string(fop_to_string[node->operation.op_type]) +
							 " wrapped only some of its parameters.");
			const int flags = impl->generate_ocl_lazy(node, name, state);
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
				const int var1 = ++variable_index;
				const int var2 = ++variable_index;
				// each parameter is read at a fraction of the offset
				IndexMap coords = state.index_map;
				const IndexExpr flat = state.flatten(
					coords, node->operation.shape, node->operation.dimensions);
				const string offset = state.to_code(flat);
				const auto scaled = [&](FGraphNode *pred, size_t iv) {
					IndexExpr e = flat;
					if (iv != 1) {
						const long hi = flat.bound.second / (long)iv;
						e.a = 0;
						e.exprs = {
							{1, state.declare_coord(
									state.index_map.current_depth,
									state.type_of({0, hi}),
									"(" + offset + ") / " + to_string(iv), 0,
									hi)}};
						e.derive_bound(state.index_atoms);
					}
					IndexMap map =
						split_atoms(e, pred->operation.shape,
									pred->operation.dimensions, state);
					map.current_depth = state.index_map.current_depth;
					return map;
				};
				todo.push_front({node->predecessors[1], "v" + to_string(var2),
								 false, scaled(node->predecessors[1], iv2)});
				todo.push_front({node->predecessors[0], "v" + to_string(var1),
								 false, scaled(node->predecessors[0], iv1)});
			}
		}
#ifdef FLINT_DEBUG
		code.prepend("// " + opstr + "(" +
					 to_string(state.index_map.current_depth) + " )\n");
#endif
		// push predecessors dfs
		if (push_pred) {
			const bool pred_same_index = same_index && passesIndexOn(node);
			// the operation has no mapping of its own yet, so it calculated the
			// index of its predecessors into `index`. They pick it up from
			// there, split into their own coordinates.
			for (int i = 0; i < node->num_predecessor; i++) {
				string parname = "v" + to_string(++variable_index);
				const FOperation pred = node->predecessors[i]->operation;
				if (state.pred_index_maps.empty()) {
					flogging(F_ERROR,
							 string(fop_to_string[node->operation.op_type]) +
								 " did not map its parameters.");
					break;
				}
				const IndexMap &pred_map = state.pred_index_maps[i];
				// the queue is processed from the front, so what is pushed
				// first ends up first in the code
				if (!state.pred_wrappers.empty() &&
					!state.pred_wrappers[i].first.empty())
					state.insert_code(state.pred_wrappers[i].first);
				todo.push_front({node->predecessors[i], parname,
								 pred_same_index, pred_map});
				if (!state.pred_wrappers.empty() &&
					!state.pred_wrappers[i].second.empty())
					state.insert_code(state.pred_wrappers[i].second);
			}
		}
	}
	// no scope is open any more -> pending
	state.insert_remaining_decls();
	code.prepend(index_decl);
	{
		// flatten the ids back to the index of the element
		Twine index_expr = itype + " index = get_global_id(0)";
		size_t stride = 1;
		for (int i = 1; i < range.dims; i++) {
			stride *= range.size[i - 1];
			index_expr.append(" + get_global_id(" + to_string(i) + ") * " +
							  to_string(stride));
		}
		index_expr.append(";\n");
		code.prepend(index_expr);
	}
	scalars = std::move(state.scalars);
	return code;
}
