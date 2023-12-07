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
	// we use breadth first search to traverse to operation graph
	list<tuple<FGraphNode *, string>> &todo = state.todo;
	// some operations work on the parameters, allow them to keep track
	unordered_map<FGraphNode *, std::string> &assigned_params =
		state.assigned_params;
	unsigned int &variable_index = state.variable_index;
	string code = "";
	// indexing logic (we save the old index in old_index$i to restore it)
	unsigned int &num_indices = state.num_indices;
	;
	todo.push_front({node, "v0"});
	while (!todo.empty()) {
		// take from queue
		const auto [node, name] = todo.front();
		todo.pop_front();
		string index_defs = "";
		// used to insert code at a specific place
		if (!node) {
			code = name + code;
			continue;
		}
		// cash var
		string type = typeString(node->operation.data_type);
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
			code = "const " + type + " " + name + " = " +
				   assigned_params[node] + "[index%" + to_string(num_entries) +
				   "];\n" + code;
		} else
			switch (node->operation.op_type) {
			// Binary Operators
			case FADD:
			case FSUB:
			case FDIV:
			case FMUL: {
				inverse_broadcasting = true;
				// size of current variable has to be equal to the size of one
				// opperand, the other one is at least smaller but not larger
				char op = '\0';
				switch (node->operation.op_type) {
				case FADD:
					op = '+';
					break;
				case FSUB:
					op = '-';
					break;
				case FDIV:
					op = '/';
					break;
				case FMUL:
					op = '*';
					break;
				default:
					break; // shut up compiler
				}
				code = "const " + type + " " + name + " = v" +
					   to_string(variable_index + 1) + " " + op + " v" +
					   to_string(variable_index + 2) + ";\n" + code;
				break;
			}
			case FPOW: {
				inverse_broadcasting = true;
				const FOperation x = node->predecessors[0]->operation;
				const FOperation y = node->predecessors[1]->operation;
				if ((x.data_type == F_FLOAT32 || x.data_type == F_FLOAT64) &&
					(y.data_type == F_FLOAT32 || y.data_type == F_FLOAT64))
					code = "const " + type + " " + name + " = pow((" + type +
						   ")v" + to_string(variable_index + 1) + ", (" + type +
						   ")v" + to_string(variable_index + 2) + ");\n" + code;
				else if (x.data_type == F_INT64 &&
						 (y.data_type == F_INT32 || y.data_type == F_INT64))
					code = "const " + type + " " + name +
						   " = (long)pown((double)v" +
						   to_string(variable_index + 1) + ", (int)v" +
						   to_string(variable_index + 2) + ");\n" + code;
				else if (x.data_type == F_INT32 &&
						 (y.data_type == F_INT32 || y.data_type == F_INT64))
					code = "const " + type + " " + name +
						   " = (int)pown((float)v" +
						   to_string(variable_index + 1) + ", (int)v" +
						   to_string(variable_index + 2) + ");\n" + code;
				else
					code = "const " + type + " " + name + " = pow((double)v" +
						   to_string(variable_index + 1) + ", (double)v" +
						   to_string(variable_index + 2) + ");\n" + code;
			} break;
			case FMIN: {
				inverse_broadcasting = true;
				code = "const " + type + " " + name + " = min((" + type + ")v" +
					   to_string(variable_index + 1) + ", (" + type + ")v" +
					   to_string(variable_index + 2) + ");\n" + code;

			} break;
			case FMAX: {
				inverse_broadcasting = true;
				code = "const " + type + " " + name + " = max((" + type + ")v" +
					   to_string(variable_index + 1) + ", (" + type + ")v" +
					   to_string(variable_index + 2) + ");\n" + code;

			} break;
			case FLESS: {
				inverse_broadcasting = true;
				code = "const " + type + " " + name + " = v" +
					   to_string(variable_index + 1) + " < v" +
					   to_string(variable_index + 2) + " ? 1 : 0;\n" + code;

			} break;
			case FEQUAL: {
				inverse_broadcasting = true;
				const FOperation x = node->predecessors[0]->operation;
				const FOperation y = node->predecessors[1]->operation;
				code = "const " + type + " " + name + " = v" +
					   to_string(variable_index + 1) + " + " +
					   epsilonForType(x.data_type) + " >= v" +
					   to_string(variable_index + 2) +
					   " && "
					   "v" +
					   to_string(variable_index + 1) + " <= v" +
					   to_string(variable_index + 2) + " + " +
					   epsilonForType(y.data_type) + "? 1 : 0;\n" + code;

			} break;
			case FGREATER: {
				inverse_broadcasting = true;
				code = "const " + type + " " + name + " = v" +
					   to_string(variable_index + 1) + " > v" +
					   to_string(variable_index + 2) + " ? 1 : 0;\n" + code;

			} break;
			case FCONCAT: {
				FGraphNode *a = node->predecessors[0];
				FGraphNode *b = node->predecessors[1];
				unsigned int old_idx = num_indices++;
				unsigned int ax =
					((unsigned int *)node->operation.additional_data)[0];
				size_t acc_size_last = 1;
				for (int i = node->operation.dimensions - 2; i >= (int)ax;
					 i--) {
					acc_size_last *= node->operation.shape[i + 1];
				}
				index_defs +=
					"long old_index" + to_string(old_idx) + " = index;\n";
				std::string sx = "index / " + to_string(acc_size_last);
				std::string sc = ax > 0
									 ? "(" + sx + ") % " +
										   to_string(node->operation.shape[ax])
									 : sx;
				std::string if_em =
					sc + " < " + to_string(a->operation.shape[ax]);
				index_defs +=
					"index = " + if_em +
					" ? "
					"(" +
					sx + " / " + to_string(node->operation.shape[ax]) + ") * " +
					to_string(acc_size_last * a->operation.shape[ax]) + " + (" +
					sc + ") * " + to_string(acc_size_last) + " + (index % " +
					to_string(acc_size_last) + "): (" + sx + " / " +
					to_string(node->operation.shape[ax]) + ") * " +
					to_string(acc_size_last * b->operation.shape[ax]) +
					" + ((" + sc + ") - " + to_string(a->operation.shape[ax]) +
					") * " + to_string(acc_size_last) + " + (index % " +
					to_string(acc_size_last) + ");\n";
				code = "index = old_index" + to_string(old_idx) + ";\n" + type +
					   " " + name + " = " + if_em + " ? v" +
					   to_string(variable_index + 1) + " : v" +
					   to_string(variable_index + 2) + ";\n" + code;
			} break;
			case FGEN_RANDOM: {
				double seed = ((double *)node->operation.additional_data)[0];
				code = type + " " + name + " = 0;\n{\n " + name +
					   " = sin(index + " + std::to_string(seed) +
					   ") * 43758.5453123;\n " + name + " = min(" + name +
					   " - floor(" + name +
					   "), 0.99999);\n"
					   "}\n" +
					   code;
			} break;
			case FGEN_ARANGE: {
				unsigned int ax =
					((unsigned int *)node->operation.additional_data)[0];
				size_t acc_sizes_ax = 1;
				for (unsigned int i = ax + 1; i < node->operation.dimensions;
					 i++)
					acc_sizes_ax *= node->operation.shape[i];
				code = "const " + type + " " + name + " = (index/" +
					   to_string(acc_sizes_ax) + ")%" +
					   to_string(node->operation.shape[ax]) + ";\n" + code;
			} break;
			case FGRADIENT_POOLING_MAX: {
				string par1, par2, par3;
				push_pred = false;
				FGraphNode *gnp3 = node->predecessors[2];
				FGraphNode *gnp2 = node->predecessors[1];
				FGraphNode *gnp1 = node->predecessors[0];
				if (assigned_params.find(gnp1) != assigned_params.end()) {
					par1 = assigned_params[gnp1];
				} else {
					par1 = "P" + to_string(assigned_params.size());
					assigned_params.insert({gnp1, par1});
					parameters.push_back({gnp1, par1});
				}
				if (assigned_params.find(gnp2) != assigned_params.end()) {
					par2 = assigned_params[gnp2];
				} else {
					par2 = "P" + to_string(assigned_params.size());
					assigned_params.insert({gnp2, par2});
					parameters.push_back({gnp2, par2});
				}
				if (assigned_params.find(gnp3) != assigned_params.end()) {
					par3 = assigned_params[gnp3];
				} else {
					par3 = "P" + to_string(assigned_params.size());
					assigned_params.insert({gnp3, par3});
					parameters.push_back({gnp3, par3});
				}
				const FOperation op = node->operation;
				const FSlidingWindow *window =
					(FSlidingWindow *)gnp1->operation.additional_data;
				const FOperation a = gnp2->operation;
				const FOperation image = gnp3->operation;
				const unsigned int *steps = window->step;
				// calculate accumulated sizes for result (pred), kernel and a
				// (adjacent)
				std::vector<size_t> acc_sizes = calcAccSizes(a);
				std::vector<size_t> acc_sizes_pred = calcAccSizes(op);
				acc_sizes[op.dimensions - 2] = 1;
				std::vector<size_t> acc_sizes_kernel(op.dimensions);
				acc_sizes_kernel[acc_sizes_kernel.size() - 1] = 1;
				acc_sizes_kernel[acc_sizes_kernel.size() - 2] =
					op.shape[op.dimensions - 1];
				for (int i = acc_sizes_kernel.size() - 3; i >= 0; i--)
					acc_sizes_kernel[i] =
						window->size[i + 1] * acc_sizes_kernel[i + 1];
				// accumulations of overlapping elements (kernel overlapping
				// itself)
				std::vector<size_t> acc_overlapping(op.dimensions - 1);
				acc_overlapping[acc_overlapping.size() - 1] = 1;
				for (int i = acc_overlapping.size() - 2; i >= 0; i--) {
					acc_overlapping[i] =
						std::max(1l,
								 (long)std::ceil((double)window->size[i + 1] /
												 (double)steps[i + 1])) *
						acc_overlapping[i + 1];
				}
				// First dimension overlap
				const size_t overlapping =
					std::max(1l, (long)std::ceil((double)window->size[0] /
												 (double)steps[0])) *
					acc_overlapping[0];
				string convc = type + " " + name + " = 0;\n{";
				convc += "int in_steps = 1, started_counting = 0;\n"
						 "long keri = 0, adji = 0;\n";
				for (int d = 0; d < op.dimensions - 1; d++) {
					convc += "if(in_steps){\nlong di = (";
					if (d == 0)
						convc += "index";
					else
						convc += "index%" + to_string(acc_sizes_pred[d - 1]);
					convc += ") / " + to_string(acc_sizes_pred[d]) +
							 ";\n"
							 "long ki = di - (di / " +
							 to_string(steps[d]) + ")*" + to_string(steps[d]) +
							 ";\n"
							 "if (ki >= " +
							 to_string(window->size[d]) +
							 ") {"
							 " in_steps = 0; }\n"
							 "keri += ki * " +
							 to_string(acc_sizes_kernel[d]) +
							 ";\n"
							 "adji += (long)ceil(max(0l, di - " +
							 to_string(window->size[d] - 1) + ") / (double)" +
							 to_string(steps[d]) + ") * " +
							 to_string(acc_sizes[d]) + ";\n}\n";
				}
				convc += "if(in_steps){\n long actual_overlapping = 0;\n keri "
						 "+= index % " +
						 to_string(op.shape[op.dimensions - 1]) +
						 ";\n for(long o = 0; o < " + to_string(overlapping) +
						 "; o++){\n  int skip_kernel = 0;\n "
						 " long adjo = "
						 "0;\n";
				for (int d = 0; d < op.dimensions - 1; d++) {
					convc += "  if(!skip_kernel){\n   const long di = (";
					if (d == 0)
						convc += "index";
					else
						convc += "index%" + to_string(acc_sizes_pred[d - 1]);
					convc += ")/" + to_string(acc_sizes_pred[d]) +
							 ";\n"
							 "   const long io = (";
					if (d == 0)
						convc += "o";
					else
						convc += "o%" + to_string(acc_overlapping[d - 1]);
					convc += ")/" + to_string(acc_overlapping[d]) +
							 ";\n"
							 "   const long ao = (";
					if (d == 0)
						convc += "actual_overlapping";
					else
						convc += "actual_overlapping%" +
								 to_string(acc_overlapping[d - 1]);
					convc += ")/" + to_string(acc_overlapping[d]) +
							 ";\n"
							 "   const long ki = (";
					if (d == 0)
						convc += "keri";
					else
						convc += "keri%" + to_string(acc_sizes_kernel[d - 1]);
					convc += ")/" + to_string(acc_sizes_kernel[d]) +
							 ";\n"
							 "   if(di + " +
							 to_string(window->size[d]) + " - (ki + io * " +
							 to_string(steps[d]) + ") > " +
							 to_string(op.shape[d]) +
							 "){\n"
							 "    if(!started_counting) actual_overlapping--;\n"
							 "    skip_kernel = true;\n"
							 "   }else if(ki + io * " +
							 to_string(steps[d]) +
							 " >= " + to_string(window->size[d]) +
							 " || di < ki + io * " + to_string(steps[d]) +
							 "){\n"
							 "    skip_kernel = true;\n"
							 "   }\n"
							 "   adjo += ao * " +
							 to_string(acc_sizes[d]) + ";\n  }\n";
				}
				convc += "  const int equal = " + par3 + "[index] == " + par1 +
						 "[adjo + adji];\n"
						 "  if(!skip_kernel && equal){\n"
						 "   started_counting = true;\n"
						 "   " +
						 name + " += " + par2 +
						 "[adji + adjo];\n"
						 " }\n"
						 " actual_overlapping++;\n}\n}\n}\n";
				code = convc + code;
			} break;
			case FGRADIENT_CONVOLVE1: {
				string par1, par2;
				push_pred = false;
				FGraphNode *gnp2 = node->predecessors[1];
				FGraphNode *gnp1 = node->predecessors[0];
				if (assigned_params.find(gnp1) != assigned_params.end()) {
					par1 = assigned_params[gnp1];
				} else {
					par1 = "P" + to_string(assigned_params.size());
					assigned_params.insert({gnp1, par1});
					parameters.push_back({gnp1, par1});
				}
				if (assigned_params.find(gnp2) != assigned_params.end()) {
					par2 = assigned_params[gnp2];
				} else {
					par2 = "P" + to_string(assigned_params.size());
					assigned_params.insert({gnp2, par2});
					parameters.push_back({gnp2, par2});
				}
				const FOperation op = node->operation;
				const FOperation kernel = gnp1->operation, a = gnp2->operation;
				const unsigned int *steps = (unsigned int *)op.additional_data;
				// calculate accumulated sizes for result (pred), kernel and a
				// (adjacent)
				std::vector<size_t> acc_sizes = calcAccSizes(a);
				std::vector<size_t> acc_sizes_pred = calcAccSizes(op);
				std::vector<size_t> acc_sizes_kernel = calcAccSizes(kernel);
				acc_sizes[op.dimensions - 2] = 1;
				size_t kernel_num_elems = kernel.shape[op.dimensions - 1];
				size_t a_num_elems = 1;
				for (long d = a.dimensions - 1; d >= 0; d--)
					a_num_elems *= a.shape[d];
				for (long d = op.dimensions - 2; d >= 0; d--)
					kernel_num_elems *= kernel.shape[d];
				// accumulations of overlapping elements (kernel overlapping
				// itself)
				std::vector<size_t> acc_overlapping(op.dimensions - 1);
				acc_overlapping[acc_overlapping.size() - 1] = 1;
				for (int i = acc_overlapping.size() - 2; i >= 0; i--) {
					acc_overlapping[i] =
						std::max(1l,
								 (long)std::ceil((double)kernel.shape[i + 1] /
												 (double)steps[i + 1])) *
						acc_overlapping[i + 1];
				}
				// First dimension overlap
				const size_t overlapping =
					std::max(1l, (long)std::ceil((double)kernel.shape[0] /
												 (double)steps[0])) *
					acc_overlapping[0];
				string convc = type + " " + name + " = 0;\n{";
				convc += "int in_steps = 1, started_counting = 0;\n"
						 "long keri = 0, adji = 0;\n";
				for (int d = 0; d < op.dimensions - 1; d++) {
					convc += "if(in_steps){\nlong di = (";
					if (d == 0)
						convc += "index";
					else
						convc += "index%" + to_string(acc_sizes_pred[d - 1]);
					convc += ") / " + to_string(acc_sizes_pred[d]) +
							 ";\n"
							 "long ki = di - (di / " +
							 to_string(steps[d]) + ")*" + to_string(steps[d]) +
							 ";\n"
							 "if (ki >= " +
							 to_string(kernel.shape[d]) +
							 ") {"
							 " in_steps = 0; }\n"
							 "keri += ki * " +
							 to_string(acc_sizes_kernel[d]) +
							 ";\n"
							 "adji += (long)ceil(max(0l, di - " +
							 to_string(kernel.shape[d] - 1) + ") / (double)" +
							 to_string(steps[d]) + ") * " +
							 to_string(acc_sizes[d]) + ";\n}\n";
				}
				convc += "if(in_steps){\n long actual_overlapping = 0;\n keri "
						 "+= index % " +
						 to_string(op.shape[op.dimensions - 1]) +
						 ";\n for(long o = 0; o < " + to_string(overlapping) +
						 "; o++){\n  int skip_kernel = 0;\n "
						 " long adjo = "
						 "0, kero = 0;\n";
				for (int d = 0; d < op.dimensions - 1; d++) {
					convc += "  if(!skip_kernel){\n   const long di = (";
					if (d == 0)
						convc += "index";
					else
						convc += "index%" + to_string(acc_sizes_pred[d - 1]);
					convc += ")/" + to_string(acc_sizes_pred[d]) +
							 ";\n"
							 "   const long io = (";
					if (d == 0)
						convc += "o";
					else
						convc += "o%" + to_string(acc_overlapping[d - 1]);
					convc += ")/" + to_string(acc_overlapping[d]) +
							 ";\n"
							 "   const long ao = (";
					if (d == 0)
						convc += "actual_overlapping";
					else
						convc += "actual_overlapping%" +
								 to_string(acc_overlapping[d - 1]);
					convc += ")/" + to_string(acc_overlapping[d]) +
							 ";\n"
							 "   const long ki = (";
					if (d == 0)
						convc += "keri";
					else
						convc += "keri%" + to_string(acc_sizes_kernel[d - 1]);
					convc +=
						")/" + to_string(acc_sizes_kernel[d]) +
						";\n"
						"   if(di + " +
						to_string(kernel.shape[d]) + " - (ki + io * " +
						to_string(steps[d]) + ") > " + to_string(op.shape[d]) +
						"){\n"
						"    if(!started_counting) actual_overlapping--;\n"
						"    skip_kernel = true;\n"
						"   }else if(ki + io * " +
						to_string(steps[d]) +
						" >= " + to_string(kernel.shape[d]) +
						" || di < ki + io * " + to_string(steps[d]) +
						"){\n"
						"    skip_kernel = true;\n"
						"   }\n"
						"   adjo += ao * " +
						to_string(acc_sizes[d]) +
						";\n"
						"   kero += io * " +
						to_string(steps[d] * acc_sizes_kernel[d]) + ";\n  }\n";
				}
				convc += "  if(!skip_kernel){\n"
						 "   started_counting = true;\n"
						 "   " +
						 name + " += " + par1 + "[keri + kero] * " + par2 +
						 "[adji + adjo];\n"
						 " }\n"
						 " actual_overlapping++;\n}\n}\n}\n";
				code = convc + code;
			} break;
			case FCONVOLVE: {
				string par1, par2;
				push_pred = false;
				FGraphNode *gnp1 = node->predecessors[0],
						   *gnp2 = node->predecessors[1];
				const bool multiple_filter =
					gnp2->operation.dimensions != gnp1->operation.dimensions;
				// we ignore the value assignment of the parameters since we
				// have to access the arrays directly parameter 1
				if (assigned_params.find(gnp1) != assigned_params.end()) {
					par1 = assigned_params[gnp1];
				} else {
					par1 = "P" + to_string(assigned_params.size());
					assigned_params.insert({gnp1, par1});
					parameters.push_back({gnp1, par1});
				}
				// parameter 2
				if (assigned_params.find(gnp2) != assigned_params.end()) {
					par2 = assigned_params[gnp2];
				} else {
					par2 = "P" + to_string(assigned_params.size());
					assigned_params.insert({gnp2, par2});
					parameters.push_back({gnp2, par2});
				}
				const FOperation op = node->operation;
				const FOperation pred = gnp1->operation,
								 kernel = gnp2->operation;
				unsigned int *steps = (unsigned int *)op.additional_data;
				vector<size_t> acc_sizes = calcAccSizes(op);
				vector<size_t> acc_sizes_pred = calcAccSizes(pred);
				vector<size_t> acc_sizes_kernel = calcAccSizes(kernel);
				size_t kernel_num_elems = kernel.shape[acc_sizes.size()];
				size_t pred_num_elems =
					multiple_filter ? 1 : pred.shape[acc_sizes.size()];
				for (long d = acc_sizes.size() - 1; d >= 0; d--) {
					pred_num_elems *= pred.shape[d];
					if (d != 0 || !multiple_filter) // since kernel.shape[0] is
													// the dimension of filters
						kernel_num_elems *= kernel.shape[d];
				}
				string conv_code = type + " " + name + " = 0;\n{\nlong j = 0";
				for (unsigned int d = 0;
					 d < (multiple_filter ? op.dimensions - 1 : op.dimensions);
					 d++)
					conv_code +=
						" + (" +
						(d == 0 ? string("index")
								: "index % " + to_string(acc_sizes[d - 1])) +
						" / " + to_string(acc_sizes[d]) + ") * " +
						to_string(steps[d] * acc_sizes_pred[d]);
				conv_code +=
					";\nlong kernel_offset = " +
					(multiple_filter
						 ? string("(index % " +
								  to_string(acc_sizes[op.dimensions - 2]) +
								  ") / " +
								  to_string(acc_sizes[op.dimensions - 1]) +
								  " * " + to_string(kernel_num_elems))
						 : string("0")) +
					";\n" + typeString(op.data_type) +
					" res = 0;\n"
					"for(long k = 0; k < " +
					to_string(kernel_num_elems) +
					"; k++){\n"
					" long o = 0;\n";
				const unsigned int last_dim = multiple_filter
												  ? acc_sizes_kernel.size() - 1
												  : acc_sizes_kernel.size();
				for (unsigned int d = 0; d < last_dim; d++) {
					const unsigned int kn_d = multiple_filter ? d + 1 : d;
					conv_code +=
						"{\nconst long di = " +
						(d == last_dim - 1
							 ? "0"
							 : (d == 0 ? string("index")
									   : "index % " +
											 to_string(acc_sizes[d - 1])) +
								   " / " + to_string(acc_sizes[d])) +
						";\n"
						"const long dk = " +
						(kn_d == 0
							 ? string("k")
							 : "k % " + to_string(acc_sizes_kernel[kn_d - 1])) +
						"/ " + to_string(acc_sizes_kernel[kn_d]) + ";\n";
					if (d < pred.dimensions - 1) {
						conv_code += "if((di * " + to_string(steps[d]) +
									 " + dk) * " +
									 to_string(acc_sizes_pred[d]) +
									 " >= " + to_string(pred_num_elems);
						if (d > 0)
							conv_code +=
								" || (di * " + to_string(steps[d]) +
								" + dk) * " + to_string(acc_sizes_pred[d]) +
								" >= " + to_string(acc_sizes_pred[d - 1]);
						conv_code += ") continue;\n";
					}
					conv_code +=
						"o += dk * " + to_string(acc_sizes_pred[d]) + ";\n}\n";
				}
				conv_code += "res += " + par2 + "[k + kernel_offset] * " +
							 par1 + "[j + o];\n}\n" + name + " = res;\n}\n";
				code = conv_code + code;
			} break;
			case FGRADIENT_CONVOLVE2: {
				const FOperation op = node->operation;
				FGraphNode *gnp1 = node->predecessors[0],
						   *gnp2 = node->predecessors[1];
				push_pred = false;
				string par1, par2;
				int vari = variable_index;
				par1 = "v" + to_string(++variable_index);
				if (assigned_params.find(gnp2) != assigned_params.end()) {
					par2 = assigned_params[gnp2];
				} else {
					par2 = "P" + to_string(assigned_params.size());
					assigned_params.insert({gnp2, par2});
					parameters.push_back({gnp2, par2});
				}
				const FOperation pred = gnp1->operation,
								 prev_adj = gnp2->operation;
				const std::vector<size_t> acc_sizes_pred = calcAccSizes(pred);
				const std::vector<size_t> acc_sizes_kernel = calcAccSizes(op);
				const bool multifilter = op.dimensions > pred.dimensions;
				const unsigned int num_filter = multifilter ? op.shape[0] : 1;
				// like accumulated sizes for prev_adj but without filter in
				// multifilter context
				std::vector<size_t> acc_sizes_windows(
					multifilter ? prev_adj.dimensions - 1
								: prev_adj.dimensions);
				acc_sizes_windows[acc_sizes_windows.size() - 1] = 1;
				for (int i = acc_sizes_windows.size() - 2; i >= 0; i--) {
					acc_sizes_windows[i] =
						acc_sizes_windows[i + 1] * prev_adj.shape[i + 1];
				}
				// total number of windows
				const size_t windows = acc_sizes_windows[0] * prev_adj.shape[0];
				// helper variables
				const size_t num_elems_kernel =
					multifilter ? acc_sizes_kernel[0]
								: acc_sizes_kernel[0] * op.shape[0];
				const unsigned int *steps = (unsigned int *)op.additional_data;
				const std::string a_offset = "a_offset" + to_string(vari);
				const std::string w = "w" + to_string(vari);
				const std::string a = "a" + to_string(vari);
				std::string grad_code =
					type + " " + name + " = 0;\nlong " + a_offset + " = 0";

				for (int j = multifilter ? 1 : 0; j < op.dimensions; j++) {
					grad_code +=
						"+((index/" + to_string(acc_sizes_kernel[j]) + ")%" +
						to_string(op.shape[j]) + ")*" +
						to_string(acc_sizes_pred[multifilter ? j - 1 : j]);
				}
				grad_code += ";\n"
							 "for(long " +
							 w + " = 0; " + w + " < " + to_string(windows) +
							 "; " + w +
							 "++){\n"
							 " long " +
							 a + " = 0";
				for (int j = 0; j < acc_sizes_windows.size(); j++) {
					grad_code += "+((" + w + "/" +
								 to_string(acc_sizes_windows[j]) + ")%" +
								 to_string(prev_adj.shape[j]) + ")*" +
								 to_string(acc_sizes_pred[j] * steps[j]);
				}
				grad_code += ";\n";
				const std::string old_idx =
					"old_idx" + to_string(num_indices++);
				grad_code += " long " + old_idx +
							 " = index;\n"
							 " index = " +
							 a + " + " + a_offset + ";\n";
				const std::string f =
					multifilter ? old_idx + " / " + to_string(num_elems_kernel)
								: "0";
				todo.push_front({nullptr, grad_code});
				todo.push_front({gnp1, par1});
				code = " " + name + "+=" + par1 + "*" + par2 + "[" + w + " * " +
					   to_string(num_filter) + " + " + f +
					   "];\n"
					   " index = " +
					   old_idx + ";\n}\n" + code;
			} break;
			case FPOOLING_SUM:
			case FPOOLING_MAX: {
				const FOperation op = node->operation;
				const FGraphNode *gnp1 = node->predecessors[0];
				const FOperation pred = gnp1->operation;
				const FSlidingWindow *window =
					(FSlidingWindow *)op.additional_data;
				// calculate accumulated sizes for result, kernel and source
				// (pred)
				const std::vector<size_t> acc_sizes = calcAccSizes(op);
				const std::vector<size_t> acc_sizes_pred = calcAccSizes(pred);
				size_t kernel_num_elems = window->size[op.dimensions - 1];
				std::vector<size_t> acc_sizes_kernel =
					std::vector<size_t>(op.dimensions);
				acc_sizes_kernel[op.dimensions - 1] = 1;
				for (int d = op.dimensions - 2; d >= 0; d--) {
					acc_sizes_kernel[d] =
						acc_sizes_kernel[d + 1] * window->size[d + 1];
					kernel_num_elems *= window->size[d];
				}
				const std::string base_ind =
					"base_ind" + to_string(variable_index);
				std::string pooling_code =
					type + " " + name + " = " +
					(op.op_type == FPOOLING_SUM ? "0"
												: minForType(op.data_type)) +
					";\nlong " + base_ind + " = 0";
				for (int d = 0; d < op.dimensions; d++) {
					pooling_code +=
						"+" +
						(d == 0
							 ? "index"
							 : "(index%" + to_string(acc_sizes[d - 1]) + ")") +
						"/" + to_string(acc_sizes[d]) + " * " +
						to_string(window->step[d] * acc_sizes_pred[d]);
				}
				const std::string k = "k" + to_string(variable_index);
				const std::string o = "o" + to_string(variable_index);
				pooling_code += ";\n"
								"for(long " +
								k + " = 0; " + k + " < " +
								to_string(kernel_num_elems) + "; " + k +
								"++){\n"
								" long " +
								o + " = 0";
				for (int d = 0; d < op.dimensions; d++) {
					pooling_code +=
						"+" +
						(d == 0
							 ? k
							 : "(" + k + "%" +
								   to_string(acc_sizes_kernel[d - 1]) + ")") +
						"/" + to_string(acc_sizes_kernel[d]) + "*" +
						to_string(acc_sizes_pred[d]);
				}
				const std::string ld = "ld" + to_string(variable_index);
				const unsigned int old_idx = num_indices++;
				pooling_code += ";\n for(long " + ld + " = 0; " + ld + " < " +
								to_string(pred.shape[pred.dimensions - 1]) +
								"; " + ld +
								"++){\n"
								"  long old_idx" +
								to_string(old_idx) +
								" = index;\n"
								"  index = " +
								base_ind + "+" + o + "+" + ld + ";\n";
				index_defs += pooling_code;
				code = "  index = old_idx" + to_string(old_idx) + ";\n  " +
					   name +
					   (op.op_type == FPOOLING_SUM
							? " += v" + to_string(variable_index + 1)
							: " = max(" + name + ", v" +
								  to_string(variable_index + 1) + ")") +
					   ";\n }\n}" + code;
			} break;
			case FSLIDING_WINDOW: {
				const FOperation pred = node->predecessors[0]->operation;
				const FSlidingWindow *slidewin =
					(FSlidingWindow *)node->operation.additional_data;
				size_t acc_size = node->operation.shape[1];
				std::vector<size_t> acc_sizes_pred(pred.dimensions);
				std::vector<size_t> acc_sizes_win(pred.dimensions);
				std::vector<size_t> acc_sizes_rest(pred.dimensions);
				acc_sizes_pred[acc_sizes_pred.size() - 1] = 1;
				acc_sizes_win[acc_sizes_win.size() - 1] = 1;
				acc_sizes_rest[acc_sizes_win.size() - 1] = 1;
				for (int i = acc_sizes_pred.size() - 2; i >= 0; i--) {
					acc_size *= node->operation.shape[i + 2];
					acc_sizes_pred[i] =
						acc_sizes_pred[i + 1] * pred.shape[i + 1];
					acc_sizes_rest[i] =
						acc_sizes_rest[i + 1] * slidewin->size[i + 1];
					// no of windows in that dimension
					size_t window_size =
						pred.shape[i + 1] - slidewin->size[i + 1] + 1;
					window_size = window_size % slidewin->step[i + 1] == 0
									  ? window_size / slidewin->step[i + 1]
									  : window_size / slidewin->step[i + 1] + 1;
					acc_sizes_win[i] = acc_sizes_win[i + 1] * window_size;
				}
				const size_t num_elems = acc_size * node->operation.shape[0];
				const unsigned int old_idx = num_indices++;
				const std::string i = "old_index" + to_string(old_idx);
				index_defs += "long " + i +
							  " = index;\n"
							  "index = 0;\n{\n"
							  "long wi = (" +
							  i + "%" + to_string(num_elems) + ")/" +
							  to_string(acc_size) +
							  ";\n"
							  "long rest = " +
							  i + "%" + to_string(acc_size) + ";\n";
				for (int d = 0; d < pred.dimensions; d++) {
					std::string local_wi = "wi/" + to_string(acc_sizes_win[d]);
					std::string loc_base = local_wi + "*" +
										   to_string(acc_sizes_pred[d]) + "*" +
										   to_string(slidewin->step[d]);
					std::string local_ri = "rest/" +
										   to_string(acc_sizes_rest[d]) + "*" +
										   to_string(acc_sizes_pred[d]);
					index_defs += "index += " + loc_base + " + " + local_ri +
								  ";\n"
								  "wi %= " +
								  to_string(acc_sizes_win[d]) +
								  ";\n"
								  "rest %= " +
								  to_string(acc_sizes_rest[d]) + ";\n";
				}
				index_defs += "}\n";
				code = "const " + type + " " + name + " = v" +
					   to_string(variable_index + 1) +
					   ";\n"
					   "index = old_index" +
					   to_string(old_idx) + ";\n" + code;
			} break;
			case FUNSLIDE_WINDOW: {
				push_pred = false;
				FGraphNode *gnp1 = node->predecessors[0];
				const FOperation pred = gnp1->operation;
				std::string par1;
				if (assigned_params.find(gnp1) != assigned_params.end()) {
					par1 = assigned_params[gnp1];
				} else {
					par1 = "P" + to_string(assigned_params.size());
					assigned_params.insert({gnp1, par1});
					parameters.push_back({gnp1, par1});
				}
				const unsigned int *steps =
					(unsigned int *)node->operation.additional_data;
				const std::vector<size_t> acc_sizes = calcAccSizes(
					node->operation.dimensions, node->operation.shape);
				const std::vector<size_t> acc_sizes_pred =
					calcAccSizes(pred.dimensions, pred.shape);
				size_t no_windows[pred.dimensions - 1];
				for (int i = 0; i < pred.dimensions - 1; i++) {
					size_t window_size =
						node->operation.shape[i] - pred.shape[i + 1] + 1;
					no_windows[i] = window_size % steps[i] == 0
										? window_size / steps[i]
										: window_size / steps[i] + 1;
				}
				const std::vector<size_t> acc_no_windows =
					calcAccSizes(pred.dimensions - 1, no_windows);
				string local_code = type + " " + name +
									" = 0;\n"
									"{\n"
									"const long first_w = 0";
				for (int d = node->operation.dimensions - 1; d >= 0; d--) {
					local_code += " + max(0l, ((index / " +
								  to_string(acc_sizes[d]) + ") % " +
								  to_string(node->operation.shape[d]) + ") - " +
								  to_string(pred.shape[d + 1]) + " + 1) / " +
								  to_string(steps[d]) + " * " +
								  to_string(acc_no_windows[d]);
				}
				local_code += ";\nconst long last_w = 0";
				for (int d = node->operation.dimensions - 1; d >= 0; d--) {
					local_code += " + ((index / " + to_string(acc_sizes[d]) +
								  ") % " + to_string(node->operation.shape[d]) +
								  ") / " + to_string(steps[d]) + " * " +
								  to_string(acc_no_windows[d]);
				}
				local_code += ";\nfor(long w=first_w;w<=last_w;){\n"
							  " bool contained = true;\n"
							  " long wi = 0;\n"
							  " long wpp = 0;\n";
				for (int d = node->operation.dimensions - 1; d >= 0; d--) {
					local_code += " {\n"
								  "  const long w_start=((w/" +
								  to_string(acc_no_windows[d]) + ")%" +
								  to_string(no_windows[d]) + ")*" +
								  to_string(steps[d]) +
								  ";\n"
								  "  const long id=(index/" +
								  to_string(acc_sizes[d]) + ")%" +
								  to_string(node->operation.shape[d]) +
								  ";\n"
								  "  if(id>=w_start && id<w_start+" +
								  to_string(pred.shape[d + 1]) +
								  ")\n"
								  "   wi+=(id-w_start)*" +
								  to_string(acc_sizes_pred[d + 1]) +
								  ";\n"
								  "  else{\n"
								  "   contained = false;\n"
								  "   wpp += " +
								  to_string(acc_no_windows[d]) +
								  ";\n"
								  "  }\n"
								  " }\n";
				}
				local_code += " if(contained) {"
							  "  " +
							  name + "+=" + par1 + "[wi+w*" +
							  to_string(acc_sizes_pred[0]) +
							  "];\n"
							  "  wpp = 1;\n}\n"
							  " w += wpp;\n"
							  "}\n"
							  "}";
				code = local_code + code;
			} break;
			case FMATMUL: {
				string par1, par2;
				push_pred = false;
				FGraphNode *gnp1 = node->predecessors[0],
						   *gnp2 = node->predecessors[1];
				// we ignore the value assignment of the parameters since we
				// have to access the arrays directly parameter 1
				if (assigned_params.find(gnp1) != assigned_params.end()) {
					par1 = assigned_params[gnp1];
				} else {
					par1 = "P" + to_string(assigned_params.size());
					assigned_params.insert({gnp1, par1});
					parameters.push_back({gnp1, par1});
				}
				// parameter 2
				if (assigned_params.find(gnp2) != assigned_params.end()) {
					par2 = assigned_params[gnp2];
				} else {
					par2 = "P" + to_string(assigned_params.size());
					assigned_params.insert({gnp2, par2});
					parameters.push_back({gnp2, par2});
				}
				size_t l =
					gnp1->operation.shape[gnp1->operation.dimensions - 2];
				size_t m =
					gnp1->operation.shape[gnp1->operation.dimensions - 1];
				size_t n =
					gnp2->operation.shape[gnp2->operation.dimensions - 1];
				// we need to compute $name
				// indices j and k of $name
				string j =
					"((index % " + to_string(l * n) + ")/" + to_string(n) + ")";
				string k =
					"((index % " + to_string(l * n) + ")%" + to_string(n) + ")";
				// base index of matrix start of p1 and p2
				string base_p1 = "";
				if (gnp1->operation.dimensions > 2) {
					// get matrix number of index and then reproject
					base_p1 = "(index / " + to_string(l * n) + ") * " +
							  to_string(l * m);
				} else
					base_p1 = "0";
				string base_p2 = "";
				if (gnp2->operation.dimensions > 2) {
					// get matrix number of index and then reproject
					base_p2 = "(index / " + to_string(l * n) + ") * " +
							  to_string(m * n);
				} else
					base_p2 = "0";
				code = "for(int i = 0; i < " + to_string(m) +
					   "; i++){\n"
					   "  " +
					   name + " += " + par1 + "[" + base_p1 + " + " + j +
					   " * " + to_string(m) + " + i] * " + par2 + "[" +
					   base_p2 + " + i * " + to_string(n) + " + " + k +
					   "];\n}\n" + code;
				code = type + " " + name + " = 0;\n" + code;
			} break;
			case FRESHAPE:
			case FLATTEN: {
				code = "const " + type + " " + name + " = v" +
					   to_string(variable_index + 1) + ";\n" + code;
			} break;
			case FCONVERSION: {
				code = "const " + type + " " + name + " = (" + type + ")v" +
					   to_string(variable_index + 1) + ";\n" + code;
			}; break;
			case FABS: {
				std::string par_name = "v" + std::to_string(variable_index + 1);
				if (node->operation.data_type < F_FLOAT32)
					code = "const " + type + " " + name + " = abs(" + par_name +
						   ");\n" + code;
				else
					code = "const " + type + " " + name + " = " + par_name +
						   "< 0 ? -" + par_name + " : " + par_name + ";\n" +
						   code;
			} break;
			case FSQRT: {
				code = "const " + type + " " + name + " = sqrt(v" +
					   std::to_string(variable_index + 1) + ");\n" + code;
			} break;
			case FEXP: {
				code = "const " + type + " " + name + " = exp(v" +
					   std::to_string(variable_index + 1) + ");\n" + code;
			} break;
			case FSIN: {
				code = "const " + type + " " + name + " = sin(v" +
					   std::to_string(variable_index + 1) + ");\n" + code;
			} break;
			case FCOS: {
				code = "const " + type + " " + name + " = cos(v" +
					   std::to_string(variable_index + 1) + ");\n" + code;
			} break;
			case FTAN: {
				code = "const " + type + " " + name + " = tan(v" +
					   std::to_string(variable_index + 1) + ");\n" + code;
			} break;
			case FASIN: {
				code = "const " + type + " " + name + " = asin(v" +
					   std::to_string(variable_index + 1) + ");\n" + code;
			} break;
			case FACOS: {
				code = "const " + type + " " + name + " = acos(v" +
					   std::to_string(variable_index + 1) + ");\n" + code;
			} break;
			case FATAN: {
				code = "const " + type + " " + name + " = atan(v" +
					   std::to_string(variable_index + 1) + ");\n" + code;
			} break;
			case FLOG: {
				code = "const " + type + " " + name + " = log(v" +
					   std::to_string(variable_index + 1) + ");\n" + code;
			} break;
			case FLOG2: {
				code = "const " + type + " " + name + " = log2(v" +
					   std::to_string(variable_index + 1) + ");\n" + code;
			} break;
			case FLOG10: {
				code = "const " + type + " " + name + " = log10(v" +
					   std::to_string(variable_index + 1) + ");\n" + code;
			} break;
			case FNEG: {
				code = "const " + type + " " + name + " = -v" +
					   std::to_string(variable_index + 1) + ";\n" + code;
			} break;
			case FSIGN: {
				code = "const " + type + " " + name + " = v" +
					   std::to_string(variable_index + 1) + " < 0 ? -1 : 1;\n" +
					   code;
			} break;
			case FEVEN: {
				code = "const " + type + " " + name + " = v" +
					   std::to_string(variable_index + 1) +
					   " % 2 == 0 ? 1 : 0;\n" + code;
			} break;

			case FREDUCE_MIN:
			case FREDUCE_MAX:
			case FREDUCE_SUM:
			case FREDUCE_MUL: {
				FGraphNode *prev = node->predecessors[0];
				// we insert a index definition to introduce the for for our
				// predecessors
				std::string par1 = "v" + std::to_string(variable_index + 1);
				int red_dim = ((int *)node->operation.additional_data)[0];
				size_t it_dim =
					1; // iteration size <=> product of all dimensions along dim
				for (size_t d = red_dim + 1; d < prev->operation.dimensions;
					 d++)
					it_dim *= prev->operation.shape[d];
				index_defs += type + " " + name + " = ";
				size_t total_el_size = 1;
				for (int i = 0; i < prev->operation.dimensions; i++)
					total_el_size *= prev->operation.shape[i];
				switch (node->operation.op_type) {
				case FREDUCE_SUM:
					index_defs += "0";
					break;
				case FREDUCE_MUL:
					index_defs += "1";
					break;
				case FREDUCE_MIN:
					index_defs += maxForType(node->operation.data_type);
					break;
				case FREDUCE_MAX:
					index_defs += minForType(node->operation.data_type);
					break;
				default:
					break;
				}
				const std::string itv = "i" + to_string(variable_index);
				const unsigned int old_idx = num_indices++;
				index_defs +=
					";\nlong old_idx" + to_string(old_idx) +
					" = index;\n"
					"for(long " +
					itv + " = 0; " + itv + " < " +
					to_string(prev->operation.shape[red_dim]) + "; " + itv +
					"++){\n"
					"index = ((old_idx" +
					to_string(old_idx) + " / " + to_string(it_dim) + ") * " +

					to_string(it_dim) + " * " +
					to_string(prev->operation.shape[red_dim]) + " + (old_idx" +
					to_string(old_idx) + " % " + to_string(it_dim) + ") + " +
					itv + " * " + to_string(it_dim) + ") % " +
					to_string(total_el_size) + ";\n";
				std::string reduce_code = "";
				switch (node->operation.op_type) {
				case FREDUCE_SUM:
					reduce_code += " " + name + " += " + par1;
					break;
				case FREDUCE_MUL:
					reduce_code += " " + name + " *= " + par1;
					break;
				case FREDUCE_MIN:
					reduce_code +=
						" " + name + " = min(" + name + ", " + par1 + ")";
					break;
				case FREDUCE_MAX:
					reduce_code +=
						" " + name + " = max(" + name + ", " + par1 + ")";
					break;
				default:
					break;
				}
				reduce_code +=
					";\n}\nindex = old_idx" + to_string(old_idx) + ";\n";
				code = reduce_code + code;
			} break;
			case FSLICE: {
				FOperation pred = node->predecessors[0]->operation;
				FSlice *slice = (FSlice *)node->operation.additional_data;
				unsigned int old_idx = num_indices++;
				index_defs +=
					"int old_index" + to_string(old_idx) + " = index;\n";
				// flattened shape data
				std::vector<size_t> acc_sizes(node->operation.dimensions);
				std::vector<size_t> acc_sizes_pred(acc_sizes.size());
				for (long d = node->operation.dimensions - 1; d >= 0; d--) {
					if (d == node->operation.dimensions - 1) {
						acc_sizes[d] = 1;
						acc_sizes_pred[d] = 1;
					} else {
						acc_sizes_pred[d] =
							acc_sizes_pred[d + 1] * pred.shape[d + 1];
						acc_sizes[d] =
							acc_sizes[d + 1] * node->operation.shape[d + 1];
					}
				}
				// calculate start
				size_t start = 0;
				std::vector<long> step(node->operation.dimensions);
				for (long d = 0; d < step.size(); d++) {
					start += slice->start[d] * acc_sizes_pred[d];
				}
				index_defs += "index = (" + to_string(start);
				// accumulate index
				for (long d = 0; d < node->operation.dimensions; d++) {
					index_defs +=
						" + (((" +
						(d == 0 ? string("index")
								: string("index %" +
										 to_string(acc_sizes[d - 1]))) +
						") / " + to_string(acc_sizes[d]) + ") % " +
						to_string(node->operation.shape[d]) + ") * " +
						to_string(slice->step[d] * (long)acc_sizes_pred[d]);
				}
				index_defs += ") ;\n";
				code = "index = old_index" + to_string(old_idx) + ";\n" + code;
				code = "const " + type + " " + name + " = v" +
					   to_string(variable_index + 1) + ";\n" + code;
			} break;
			case FEXTEND: {
				const FOperation pred = node->predecessors[0]->operation;
				FExtend *extend = (FExtend *)node->operation.additional_data;
				unsigned int old_idx = num_indices++;
				index_defs +=
					"int old_index" + to_string(old_idx) + " = index;\n";
				// flattened shape data
				std::vector<size_t> acc_sizes(node->operation.dimensions);
				std::vector<size_t> acc_sizes_pred(acc_sizes.size());
				for (long d = node->operation.dimensions - 1; d >= 0; d--) {
					if (d == node->operation.dimensions - 1) {
						acc_sizes[d] = 1;
						acc_sizes_pred[d] = 1;
					} else {
						acc_sizes_pred[d] =
							acc_sizes_pred[d + 1] * pred.shape[d + 1];
						acc_sizes[d] =
							acc_sizes[d + 1] * node->operation.shape[d + 1];
					}
				}
				// calculate start
				index_defs += "index = 0";
				std::string set_zero_cond = "if(";
				// accumulate index
				for (long d = 0; d < node->operation.dimensions; d++) {
					long step = extend->step[d];
					bool inv = step < 0;
					if (inv)
						step = -step;
					std::string dim_idx =
						"((" +
						(d == 0 ? string("index")
								: string("index %" +
										 to_string(acc_sizes[d - 1]))) +
						") / " + to_string(acc_sizes[d]) + " - " +
						to_string(extend->start[d]) + ") / " + to_string(step);
					if (d != 0)
						set_zero_cond += " || ";
					// if di < start
					set_zero_cond +=
						"(" +
						(d == 0 ? string("index")
								: string("index %" +
										 to_string(acc_sizes[d - 1]))) +
						") / " + to_string(acc_sizes[d]) + " < " +
						to_string(extend->start[d]);
					// if di % step != 0
					set_zero_cond +=
						" || ((" +
						(d == 0 ? string("index")
								: string("index %" +
										 to_string(acc_sizes[d - 1]))) +
						") / " + to_string(acc_sizes[d]) + " - " +
						to_string(extend->start[d]) + ") % " + to_string(step) +
						" != 0";
					// if di >= shape
					set_zero_cond +=
						" || " + dim_idx + " >= " + to_string(pred.shape[d]);

					// finish index
					if (inv)
						dim_idx = "(" + to_string(pred.shape[d]) + " - " +
								  dim_idx + " - 1)";
					index_defs +=
						" + " + dim_idx + " * " + to_string(acc_sizes_pred[d]);
				}
				index_defs += ";\nif(index < 0) index = 0;\n";
				code = set_zero_cond + ") " + name + " = 0;\n" + code;
				code = "index = old_index" + to_string(old_idx) + ";\n" + code;
				code = type + " " + name + " = v" +
					   to_string(variable_index + 1) + ";\n" + code;
			} break;
			case FREPEAT: {
				const FOperation op = node->operation;
				const FOperation pred = node->predecessors[0]->operation;
				const unsigned int old_idx = num_indices++;
				index_defs +=
					"int old_index" + to_string(old_idx) + " = index;\n";
				// add to index_defs a redefinition of index, so that we remap
				// to src data calculate number of elements per dimension entry
				// for destination and source
				std::vector<size_t> acc_sizes_d(op.dimensions);
				std::vector<size_t> acc_sizes_s(op.dimensions);
				acc_sizes_d[op.dimensions - 1] = 1;
				acc_sizes_s[op.dimensions - 1] = 1;
				for (int dim = op.dimensions - 2; dim >= 0; dim--) {
					acc_sizes_d[dim] = acc_sizes_d[dim + 1] * op.shape[dim + 1];
					acc_sizes_s[dim] =
						acc_sizes_s[dim + 1] * pred.shape[dim + 1];
				}
				// to get the index in the source array we first calculate the
				// indices and reproject
				index_defs += "{\nint working_index = index;\nindex = 0;\n";
				for (int dim = 0; dim < op.dimensions; dim++) {
					index_defs += "index += ((working_index /" +
								  to_string(acc_sizes_d[dim]) + ") % " +
								  to_string(pred.shape[dim]) + ") * " +
								  to_string(acc_sizes_s[dim]) + ";\n";
					index_defs +=
						"working_index %= " + to_string(acc_sizes_d[dim]) +
						";\n";
				}
				index_defs += "}\n";
				code = "index = old_index" + to_string(old_idx) + ";\n" + code;
				code = "const " + type + " " + name + " = v" +
					   to_string(variable_index + 1) + ";\n" + code;
			} break;
			case FTRANSPOSE: {
				const FOperation op = node->operation;
				const int *transposition = (int *)op.additional_data;
				const FOperation pred = node->predecessors[0]->operation;
				unsigned int old_idx = num_indices++;
				index_defs +=
					"long old_index" + to_string(old_idx) + " = index;\n";
				// add to index_defs a redefinition of index, so that we remap
				// to src data calculate number of elements per dimension entry
				// for destination and source
				std::vector<size_t> acc_sizes_d(op.dimensions);
				std::vector<size_t> acc_sizes_s(op.dimensions);
				acc_sizes_d[op.dimensions - 1] = 1;
				acc_sizes_s[op.dimensions - 1] = 1;
				for (int dim = op.dimensions - 2; dim >= 0; dim--) {
					acc_sizes_d[dim] = acc_sizes_d[dim + 1] * op.shape[dim + 1];
					acc_sizes_s[dim] =
						acc_sizes_s[dim + 1] * pred.shape[dim + 1];
				}
				// to get the index in the source array we first calculate the
				// indices and reproject
				index_defs += "{\nint working_index = index;\nindex = 0;\n";
				for (int dim = 0; dim < op.dimensions; dim++) {
					index_defs += "index += ((working_index /" +
								  to_string(acc_sizes_d[dim]) + ") % " +
								  to_string(op.shape[dim]) + ") * " +
								  to_string(acc_sizes_s[transposition[dim]]) +
								  ";\n";
					index_defs +=
						"working_index %= " + to_string(acc_sizes_d[dim]) +
						";\n";
				}
				index_defs += "}\n";
				code = "index = old_index" + to_string(old_idx) + ";\n" + code;
				code = "const " + type + " " + name + " = v" +
					   to_string(variable_index + 1) + ";\n" + code;
			} break;
			case FSET_INDEX: {
				FGraphNode *a = node->predecessors[0];
				FGraphNode *b = node->predecessors[1];
				FGraphNode *c = node->predecessors[2];
				const FOperation op = node->operation;
				const unsigned int axis = c->operation.dimensions - 1;
				string par1, par2, par3;
				push_pred = false;
				// index has to be a calculated parameter
				if (assigned_params.find(c) != assigned_params.end()) {
					par3 = assigned_params[c];
				} else {
					par3 = "P" + to_string(assigned_params.size());
					assigned_params.insert({c, par3});
					parameters.push_back({c, par3});
				}
				// b as well
				if (assigned_params.find(b) != assigned_params.end()) {
					par2 = assigned_params[b];
				} else {
					par2 = "P" + to_string(assigned_params.size());
					assigned_params.insert({b, par2});
					parameters.push_back({b, par2});
				}
				// a may be calculated lazily
				par1 = "v" + to_string(++variable_index);
				size_t acc_sizes_ax = 1;
				for (int i = axis + 1; i < op.dimensions; i++)
					acc_sizes_ax *= op.shape[i];
				const std::string base =
					"index / " + to_string(acc_sizes_ax * op.shape[axis]);
				const std::string rest = "index % " + to_string(acc_sizes_ax);
				const std::string axi = "(index / " + to_string(acc_sizes_ax) +
										")%" + to_string(op.shape[axis]);
				const std::string ind = "(long) " + par3 + "[index / " +
										to_string(acc_sizes_ax) + "]";
				const std::string base_ind =
					base + " * " + to_string(c->operation.shape[axis]);
				code = type + " " + name +
					   " = 0;\n"
					   "{const long base_ind = " +
					   base_ind +
					   ";\n"
					   " const long axi = " +
					   axi +
					   ";\n"
					   " const long rest = " +
					   rest +
					   ";\n"
					   "int found_something = false;\n"
					   " for(long j = 0; j < " +
					   to_string(c->operation.shape[axis]) +
					   "; j++){\n"
					   "  const long ind = " +
					   par3 +
					   "[base_ind + j];\n"
					   "  if(ind == axi) {\n   " +
					   name + " += " + par2 + "[(base_ind + j) * " +
					   to_string(acc_sizes_ax) +
					   " + rest];\n"
					   "   found_something = true;\n"
					   "  }\n"
					   " }\n"
					   " if(!found_something) " +
					   name + " = " + par1 +
					   ";\n"
					   "}\n" +
					   code;
				todo.push_front({a, par1});
			} break;
			case FINDEX: {
				FGraphNode *a = node->predecessors[0];
				FGraphNode *b = node->predecessors[1];
				const FOperation op = node->operation;
				const unsigned int axis = b->operation.dimensions - 1;
				string par1, par2;
				push_pred = false;
				par1 = "v" + to_string(++variable_index);
				par2 = "v" + to_string(++variable_index);
				size_t acc_sizes_ax = 1;
				for (int i = axis + 1; i < op.dimensions; i++)
					acc_sizes_ax *= op.shape[i];

				const std::string base =
					"index / " + to_string(acc_sizes_ax * op.shape[axis]);
				const std::string rest = "index % " + to_string(acc_sizes_ax);
				unsigned int old_idx1 = num_indices++;
				unsigned int old_idx2 = num_indices++;
				std::string local_index_def1 =
					"index = old_index" + to_string(old_idx2) +
					";\nlong old_index" + to_string(old_idx1) + " = index;\n";
				local_index_def1 +=
					"index = " + base + " * " +
					to_string(acc_sizes_ax * a->operation.shape[axis]) + " + " +
					par2 + " * " + to_string(acc_sizes_ax) + " + (" + rest +
					");\n";
				code = "index = old_index" + to_string(old_idx1) + ";\n" +
					   type + " " + name + " = " + par1 + ";\n" + code;
				std::string local_index_def2 = "long old_index" +
											   to_string(old_idx2) +
											   " = index;\n"
											   "index /= " +
											   to_string(acc_sizes_ax) + ";\n";
				todo.push_front({nullptr, local_index_def2});
				todo.push_front({b, par2});
				todo.push_front({nullptr, local_index_def1});
				todo.push_front({a, par1});
			}
			default:
				break;
			}
		if (inverse_broadcasting) {
			// manipulate for invserse broadcasting
			size_t iv1 = 1, iv2 = 1;
			calculateDivisorForInverseBroadcasting(node->predecessors[0], iv1,
												   node->predecessors[1], iv2);
			if (iv1 != 1 || iv2 != 1) {
				push_pred = false;
				const string old_idx = "old_idx" + to_string(num_indices++);
				code = "index = " + old_idx + ";\n" + code;
				int var1 = ++variable_index;
				int var2 = ++variable_index;
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
		code = "// " + opstr + "\n" + code;
#endif
		// insert our indexing logic into the queue after the children
		if (!index_defs.empty())
			todo.push_front({nullptr, index_defs});
		// push predecessors dfs
		if (push_pred)
			for (int i = 0; i < node->num_predecessor; i++) {
				string parname = "v" + to_string(++variable_index);
				todo.push_front({node->predecessors[i], parname});
			}
	}
	code = "long index = get_global_id(0);\n" + code;
	return code;
}
