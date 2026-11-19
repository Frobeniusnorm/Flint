/* Copyright 2026 David Schwarzbeck
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
#include <flint/flint.h>
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
/** How often each node is read by the to-be-executed graph `root`./ */
static std::unordered_map<FGraphNode *, size_t>
collectInsideUses(FGraphNode *root) {
	using namespace std;
	unordered_map<FGraphNode *, size_t> uses;
	unordered_set<FGraphNode *> expanded;
	list<FGraphNode *> todo = {root};
	while (!todo.empty()) {
		FGraphNode *curr = todo.front();
		todo.pop_front();
		if (!expanded.insert(curr).second || isKernelParameter(curr))
			continue;
		for (int i = 0; i < curr->num_predecessor; i++) {
			uses[curr->predecessors[i]]++;
			todo.push_back(curr->predecessors[i]);
		}
	}
	return uses;
}
/** Copies of stored nodes with one of their dimensions moved to the end, by
 * the node they were copied from: (the dimension, the copy). Neighbouring work
 * items are executed together, if they read neighbouring entries that is one
 * memory access instead of one per work item. */
static std::unordered_map<const FGraphNode *,
						  std::vector<std::pair<int, FGraphNode *>>>
	layout_copies;
/** Larger nodes are not copied */
static const size_t MAX_LAYOUT_COPY = 1 << 20;
/** How often each entry has to be read for a copy to pay off */
static const size_t MIN_LAYOUT_READS = 64;
void freeLayoutCopies(const FGraphNode *node) {
	const auto copies = layout_copies.find(node);
	if (copies == layout_copies.end())
		return;
	for (const auto &[dim, copy] : copies->second)
		if (--copy->reference_counter == 0)
			fFreeGraph(copy);
	layout_copies.erase(copies);
}
/** The copy of `node` with dimension `dim` moved to the end */
static FGraphNode *layoutCopy(FGraphNode *node, int dim) {
	auto &copies = layout_copies[node];
	for (const auto &[d, copy] : copies)
		if (d == dim)
			return copy;
	std::vector<int> order;
	for (int i = 0; i < node->operation.dimensions; i++)
		if (i != dim)
			order.push_back(i);
	order.push_back(dim);
	// the copy may not hold the node alive, so it is no part of a gradient
	const bool in_context = fIsGradientContext();
	if (in_context)
		fStopGradientContext();
	FGraphNode *copy =
		fOptimizeMemory(fExecuteGraph(ftranspose(node, order.data())));
	if (in_context)
		fStartGradientContext();
	copy->reference_counter++;
	layout_copies[node].push_back({dim, copy});
	return copy;
}
/** The dimension of the stored `node` that should be its last one: the one
 * that changes between neighbouring work items (which is `fastest`, the atom
 * of the first launch id), if their entries are more than a cache line apart.
 * -1 if the layout is fine or a copy would not pay off. */
static int layoutDimension(const FGraphNode *node, const IndexCode &state,
						   size_t root_entries, int fastest) {
	const FOperation &op = node->operation;
	if (fastest < 0 || op.dimensions < 2 ||
		state.index_map.expr_per_dim.size() != (size_t)op.dimensions)
		return -1;
	size_t entries = 1, reads = root_entries;
	int dim = -1;
	std::unordered_set<int> loops;
	for (int i = 0; i < op.dimensions; i++) {
		entries *= op.shape[i];
		for (const auto &[c, atom] : state.index_map.expr_per_dim[i].exprs) {
			if (atom == fastest)
				dim = i;
			// each loop the address depends on reads the node again
			const IndexAtom &ia = state.index_atoms[atom];
			if (ia.depth > 0 && loops.insert(atom).second)
				reads *= ia.bound.second - ia.bound.first + 1;
		}
	}
	if (dim < 0 || dim == op.dimensions - 1 || entries > MAX_LAYOUT_COPY ||
		reads < MIN_LAYOUT_READS * entries)
		return -1;
	size_t stride = 1;
	for (int i = dim + 1; i < op.dimensions; i++)
		stride *= op.shape[i];
	return stride >= 16 ? dim : -1;
}
std::string
generateCode(FGraphNode *root_node,
			 std::list<std::pair<FGraphNode *, std::string>> &parameters,
			 std::vector<std::pair<FType, double>> &scalars,
			 std::list<std::pair<FGraphNode *, std::string>> &imm_results) {
	using namespace std;
	const LaunchRange range = launchRange(root_node);
	OCLLazyCodegenState state;
	state.parameters = &parameters;
	state.code = {};
	state.index_type =
		maximumTensorSize(root_node) < (1l << 31) ? "int" : "long";
	// the coordinates of the result, seeded from the launch ids exactly like
	// `launchRange` assigns them: the two innermost dimensions that are larger
	// than one get an id of their own, all remaining ones share the third
	Twine index_decl;
	state.index_map.expr_per_dim.resize(root_node->operation.dimensions);
	{
		int found = 0;
		size_t factor = 1;
		for (int i = root_node->operation.dimensions - 1; i >= 0; i--) {
			IndexExpr &coord = state.index_map.expr_per_dim[i];
			coord.a = 0;
			const size_t size = root_node->operation.shape[i];
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
	// the atom of the first launch id, neighbouring work items differ in it
	const int fastest = state.index_atoms.empty() ? -1 : 0;
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
	collectReusable(root_node, users, distance);
	unordered_map<FGraphNode *, string> reused;
	// nodes that are read outside of this kernel too are written into a
	// result of their own while they are calculated
	unordered_map<FGraphNode *, size_t> inside_uses =
		collectInsideUses(root_node);
	unordered_set<FGraphNode *> stored;
	size_t root_entries = 1;
	for (int i = 0; i < root_node->operation.dimensions; i++)
		root_entries *= root_node->operation.shape[i];
	// the node currently generated as a reused definition
	FGraphNode *defining = nullptr;
	// the coordinates of the result
	const IndexMap root_map = state.index_map;
	todo.push_front({root_node, "v0", true, state.index_map});
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
				state.flatten(root_coords, root_node->operation.shape,
							  root_node->operation.dimensions),
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
		// store intermediate result if referenced outside and evaluated with
		// the basic index of the kernel (else we evaluate it multiple times or
		// not exhaustively).
		const FOperationType optype = node->operation.op_type;
		size_t num_entries = 1;
		for (int i = 0; i < node->operation.dimensions; i++)
			num_entries *= node->operation.shape[i];
		if (node != root_node && !node->result_data && optype != FSTORE &&
			optype != FGEN_CONSTANT && optype != FGEN_ARANGE &&
			inside_uses[node] < (size_t)node->reference_counter && same_index &&
			index_map.current_depth == 0 && num_entries == root_entries &&
			stored.insert(node).second) {
			const std::string immediate = "IR" + to_string(stored.size() - 1);
			imm_results.push_back({node, immediate});
			const IndexExpr flat =
				state.flatten(state.index_map, node->operation.shape,
							  node->operation.dimensions);
			code.prepend(immediate + "[" + state.to_code(flat) + "] = " + name +
						 ";\n");
		}
		// write code
		const string opstr = string(fop_to_string[node->operation.op_type]);
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
			// read from a copy in another layout if that one suits the work
			// items better
			FGraphNode *source = node;
			IndexMap source_map = state.index_map;
			const int last = layoutDimension(node, state, root_entries, fastest);
			if (last >= 0) {
				source = layoutCopy(node, last);
				auto &coords = source_map.expr_per_dim;
				coords.push_back(coords[last]);
				coords.erase(coords.begin() + last);
			}
			if (assigned_params.find(source) == assigned_params.end()) {
				size_t pid = assigned_params.size();
				assigned_params.insert({source, "P" + to_string(pid)});
				parameters.push_back({source, "P" + to_string(pid)});
			}
			const IndexExpr flat =
				state.flatten(source_map, source->operation.shape,
							  source->operation.dimensions);
			string address = state.to_code(flat);
			if (flat.bound.first < 0 ||
				flat.bound.second >= (long)num_entries) {
				if (address.find(' ') != string::npos && address[0] != '(')
					address = "(" + address + ")";
				address += " % " + to_string(num_entries);
			}
			code.prepend("const " + type + " " + name + " = " +
						 assigned_params[source] + "[" + address + "];\n");
		} else {
			OperationImplementation *impl =
				OperationImplementation::implementations[node->operation
															 .op_type];
			// where are the parameters read?
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
			push_pred =
				(flags & OperationImplementation::OCL_LAZY_DONT_PUSH_PREDS) ==
				0;
		}
#ifdef FLINT_DEBUG
		code.prepend("// " + opstr + "\n");
#endif
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
	{
		// the index of the element, flattened from its coordinates. Not from
		// the launch ids: the coordinates are bounded, and with them known
		// the same address is measurably faster to write to
		Twine index_expr = itype + " index = 0";
		size_t stride = 1;
		for (int i = root_node->operation.dimensions - 1; i >= 0; i--) {
			const IndexExpr &coord = root_map.expr_per_dim[i];
			if (!coord.exprs.empty())
				index_expr.append(
					" + " + state.index_atoms[coord.exprs[0].second].name +
					(stride > 1 ? " * " + to_string(stride) : ""));
			stride *= root_node->operation.shape[i];
		}
		index_expr.append(";\n");
		code.prepend(index_expr);
	}
	code.prepend(index_decl);
	scalars = std::move(state.scalars);
	return code;
}
