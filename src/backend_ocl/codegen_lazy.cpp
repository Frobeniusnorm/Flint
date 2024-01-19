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

std::string
generateCode(FGraphNode *node,
			 std::list<std::pair<FGraphNode *, std::string>> &parameters) {
	using namespace std;
	OCLLazyCodegenState state;
	state.parameters = &parameters;
	state.code = {};
	// we use breadth first search to traverse to operation graph
	list<tuple<FGraphNode *, string>> &todo = state.todo;
	// some operations work on the parameters, allow them to keep track
	unordered_map<FGraphNode *, std::string> &assigned_params =
		state.assigned_params;
	unsigned int &variable_index = state.variable_index;
	Twine &code = state.code;
	// indexing logic (we save the old index in old_index$i to restore it)
	unsigned int &num_indices = state.num_indices;
	todo.push_front({node, "v0"});
	while (!todo.empty()) {
		// take from queue
		const auto [node, name] = todo.front();
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
		// write code
		const string opstr = string(fop_to_string[node->operation.op_type]);
		bool inverse_broadcasting =
			false; // adds index manipulation code for inverse broadcasting
		// need to be outside switch to include result_data
		if (node->operation.op_type == FSTORE || node->result_data ||
			node->operation.op_type == FGEN_CONSTANT) {
			push_pred = false;
			size_t num_entries =
				node->operation.op_type == FSTORE
					? ((FStore *)node->operation.additional_data)->num_entries
					: (node->operation.op_type == FGEN_CONSTANT
						   ? 1
						   : node->result_data->num_entries);
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
			calculate_divisor_for_inverse_broadcasting(node->predecessors[0], iv1,
												   node->predecessors[1], iv2);
			if (iv1 != 1 || iv2 != 1) {
				push_pred = false;
				const string old_idx = "old_idx" + to_string(num_indices++);
				code.prepend("index = " + old_idx + ";\n");
				const int var1 = ++variable_index;
				const int var2 = ++variable_index;
				todo.push_front({nullptr, "long " + old_idx +
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
		if (push_pred)
			for (int i = 0; i < node->num_predecessor; i++) {
				string parname = "v" + to_string(++variable_index);
				todo.push_front({node->predecessors[i], parname});
			}
	}
	code.prepend("long index = get_global_id(0);\n");
	return code;
}
