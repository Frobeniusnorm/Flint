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

  This file includes the implementation of the GPU backend code generation and
  should only be included in oclimpl.cpp.
*/

#ifndef OCL_CODEGEN_HPP
#define OCL_CODEGEN_HPP
#include "../flint.h"
#include "utils.hpp"
#include <list>
#include <string>
#include <tuple>
#include <unordered_map>
static std::string
generateCode(FGraphNode *node,
             std::list<std::pair<FGraphNode *, std::string>> &parameters) {
  using namespace std;
  // we use breadth first search to traverse to operation graph
  list<tuple<FGraphNode *, string>> todo;
  // some operations work on the parameters, allow them to keep track
  unordered_map<FGraphNode *, std::string> assigned_params;
  int variable_index = 0;
  string code = "";
  // indexing logic (we save the old index in old_index$i to restore it)
  unsigned int num_indices = 0;
  todo.push_front({node, "v0"});
  while (!todo.empty()) {
    // take from queue
    const auto [node, name] = todo.front();
    todo.pop_front();
    string index_defs = "";
    if (!node) {
      code = name + code;
      continue;
    }
    bool push_pred = true;
    // write code
    string type = typeString(node->operation->data_type);
    const string opstr = string(fop_to_string[node->operation->op_type]);
    // need to be outside switch to include result_data
    if (node->operation->op_type == FSTORE || node->result_data) {
      push_pred = false;
      size_t num_entries =
          node->operation->op_type == FSTORE
              ? ((FStore *)node->operation->additional_data)->num_entries
              : node->result_data->num_entries;
      if (assigned_params.find(node) == assigned_params.end()) {
        size_t pid = assigned_params.size();
        assigned_params.insert({node, "P" + to_string(pid)});
        parameters.push_back({node, "P" + to_string(pid)});
      }
      code = type + " " + name + " = " + assigned_params[node] + "[index%" +
             to_string(num_entries) + "];\n" + code;
    } else
      switch (node->operation->op_type) {
      case FCONST:
        switch (node->operation->data_type) {
        case F_INT32: {
          FConst *actcst = (FConst *)node->operation->additional_data;
          code = type + " " + name + " = " +
                 to_string(*((int *)actcst->value)) + ";\n" + code;
        } break;
        case F_INT64: {
          FConst *actcst = (FConst *)node->operation->additional_data;
          code = type + " " + name + " = " +
                 to_string(*((long *)actcst->value)) + ";\n" + code;
        } break;
        case F_FLOAT64: {
          FConst *actcst = (FConst *)node->operation->additional_data;
          code = type + " " + name + " = " +
                 to_string(*((double *)actcst->value)) + ";\n" + code;
        } break;
        case F_FLOAT32: {
          FConst *actcst = (FConst *)node->operation->additional_data;
          code = type + " " + name + " = " +
                 to_string(*((float *)actcst->value)) + ";\n" + code;
        } break;
        }
        break;
      // Binary Operators
      case FADD:
      case FSUB:
      case FDIV:
      case FMUL: {
        // size of current variable has to be equal to the size of one opperand,
        // the other one is at least smaller but not larger
        char op = '\0';
        switch (node->operation->op_type) {
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
        code = type + " " + name + " = v" + to_string(variable_index + 1) +
               " " + op + " v" + to_string(variable_index + 2) + ";\n" + code;
        break;
      }
      case FPOW: {
        FOperation *x = node->predecessors[0]->operation;
        FOperation *y = node->predecessors[1]->operation;
        if ((x->data_type == F_FLOAT32 || x->data_type == F_FLOAT64) &&
            (y->data_type == F_FLOAT32 || y->data_type == F_FLOAT64))
          code = type + " " + name + " = pow((" + type + ")v" +
                 to_string(variable_index + 1) + ", (" + type + ")v" +
                 to_string(variable_index + 2) + ");\n" + code;
        else if (x->data_type == F_INT64 &&
                 (y->data_type == F_INT32 || y->data_type == F_INT64))
          code = type + " " + name + " = (long)pown((double)v" +
                 to_string(variable_index + 1) + ", (int)v" +
                 to_string(variable_index + 2) + ");\n" + code;
        else if (x->data_type == F_INT32 &&
                 (y->data_type == F_INT32 || y->data_type == F_INT64))
          code = type + " " + name + " = (int)pown((float)v" +
                 to_string(variable_index + 1) + ", (int)v" +
                 to_string(variable_index + 2) + ");\n" + code;
        else
          code = type + " " + name + " = pow((double)v" +
                 to_string(variable_index + 1) + ", (double)v" +
                 to_string(variable_index + 2) + ");\n" + code;
      } break;
      case FMIN: {
        code = type + " " + name + " = min((" + type + ")v" +
               to_string(variable_index + 1) + ", (" + type + ")v" +
               to_string(variable_index + 2) + ");\n" + code;

      } break;
      case FMAX: {
        code = type + " " + name + " = max((" + type + ")v" +
               to_string(variable_index + 1) + ", (" + type + ")v" +
               to_string(variable_index + 2) + ");\n" + code;

      } break;
      case FLESS: {
        code = type + " " + name + " = v" + to_string(variable_index + 1) +
               " < v" + to_string(variable_index + 2) + " ? 1 : 0;\n" + code;

      } break;
      case FEQUAL: {
        code = type + " " + name + " = v" + to_string(variable_index + 1) +
               " == v" + to_string(variable_index + 2) + " ? 1 : 0;\n" + code;

      } break;
      case FGREATER: {
        code = type + " " + name + " = v" + to_string(variable_index + 1) +
               " > v" + to_string(variable_index + 2) + " ? 1 : 0;\n" + code;

      } break;
      case FMATMUL: {
        string par1, par2;
        push_pred = false;
        FGraphNode *gnp1 = node->predecessors[0], *gnp2 = node->predecessors[1];
        // we ignore the value assignment of the parameters since we have to
        // access the arrays directly parameter 1
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
        size_t l = gnp1->operation->shape[gnp1->operation->dimensions - 2];
        size_t m = gnp1->operation->shape[gnp1->operation->dimensions - 1];
        size_t n = gnp2->operation->shape[gnp2->operation->dimensions - 1];
        // we need to compute $name
        // indices j and k of $name
        string j = "((index % " + to_string(l * n) + ")/" + to_string(n) + ")";
        string k = "((index % " + to_string(l * n) + ")%" + to_string(n) + ")";
        // base index of matrix start of p1 and p2
        string base_p1 = "";
        if (gnp1->operation->dimensions > 2) {
          // get matrix number of index and then reproject
          base_p1 = "(index / " + to_string(l * n) + ") * " + to_string(l * m);
        } else
          base_p1 = "0";
        string base_p2 = "";
        if (gnp2->operation->dimensions > 2) {
          // get matrix number of index and then reproject
          base_p2 = "(index / " + to_string(l * n) + ") * " + to_string(m * n);
        } else
          base_p2 = "0";
        code = "for(int i = 0; i < " + to_string(m) +
               "; i++){\n"
               "  " +
               name + " += " + par1 + "[" + base_p1 + " + " + j + " * " +
               to_string(m) + " + i] * " + par2 + "[" + base_p2 + " + i * " +
               to_string(n) + " + " + k + "];\n}\n" + code;
        code = type + " " + name + " = 0;\n" + code;
      } break;
      case FRESHAPE:
      case FLATTEN: {
        code = type + " " + name + " = v" + to_string(variable_index + 1) +
               ";\n" + code;
      } break;
      case FCONVERSION: {
        code = type + " " + name + " = (" + type + ")v" +
               to_string(variable_index + 1) + ";\n" + code;
      }; break;
      case FABS: {
        std::string par_name = "v" + std::to_string(variable_index + 1);
        if (node->operation->data_type < F_FLOAT32)
          code = type + " " + name + " = abs(" + par_name + ");\n" + code;
        else
          code = type + " " + name + " = " + par_name + "< 0 ? -" + par_name +
                 " : " + par_name + ";\n" + code;
      } break;
      case FLOG: {
        code = type + " " + name + " = log(v" +
               std::to_string(variable_index + 1) + ");\n" + code;
      } break;
      case FLOG2: {
        code = type + " " + name + " = log2(v" +
               std::to_string(variable_index + 1) + ");\n" + code;
      } break;
      case FLOG10: {
        code = type + " " + name + " = log10(v" +
               std::to_string(variable_index + 1) + ");\n" + code;
      } break;
      case FNEG: {
        code = type + " " + name + " = -v" +
               std::to_string(variable_index + 1) + ";\n" + code;
      } break;
      case FSIGN: {
        code = type + " " + name + " = v" + std::to_string(variable_index + 1) +
               " < 0 ? -1 : 1;\n" + code;
      } break;
      case FEVEN: {
        code = type + " " + name + " = v" + std::to_string(variable_index + 1) +
               " % 2 == 0 ? 1 : 0;\n" + code;
      } break;
      case FREDUCE_SUM:
      case FREDUCE_MUL: {
        push_pred = false;
        FGraphNode *prev = node->predecessors[0];

        int red_dim = ((int *)node->operation->additional_data)[0];
        size_t it_dim =
            1; // iteration size <=> product of all dimensions along dim
        for (size_t d = red_dim + 1; d < prev->operation->dimensions; d++)
          it_dim *= prev->operation->shape[d];
        std::string reduce_code = type + " " + name + " = ";
        reduce_code +=
            std::to_string(node->operation->op_type == FREDUCE_SUM ? 0 : 1) +
            ";\n";
        reduce_code += "for(long i = 0; i < " +
                       std::to_string(prev->operation->shape[red_dim]) +
                       "; i++){\n";
        // we ignore the value assignment of the parameters since we have to
        // access the arrays directly parameter 1
        std::string par1 = "";
        if (assigned_params.find(prev) != assigned_params.end()) {
          par1 = assigned_params[prev];
        } else {
          par1 = "P" + to_string(assigned_params.size());
          parameters.push_back({prev, par1});
          assigned_params.insert({prev, par1});
        }
        size_t total_el_size = 1;
        for (int i = 0; i < prev->operation->dimensions; i++)
          total_el_size *= prev->operation->shape[i];
        std::string reduce_index = "(index)";
        reduce_code +=
            " " + name +
            (node->operation->op_type == FREDUCE_SUM ? " += " : " *= ") + par1 +
            "[((" + reduce_index + " / " + std::to_string(it_dim) + ") * " +
            std::to_string(it_dim) + " * " +
            std::to_string(prev->operation->shape[red_dim]) + " + (" +
            reduce_index + " % " + std::to_string(it_dim) + ") + i * " +
            std::to_string(it_dim) + ") % " + to_string(total_el_size) +
            "];\n}\n";
        code = reduce_code + code;
      } break;
      case FSLICE: {
        FOperation *pred = node->predecessors[0]->operation;
        FSlice *slice = (FSlice *)node->operation->additional_data;
        unsigned int old_idx = num_indices++;
        index_defs += "int old_index" + to_string(old_idx) + " = index;\n";
        // flattened shape data
        std::vector<size_t> acc_sizes(node->operation->dimensions);
        std::vector<size_t> acc_sizes_pred(acc_sizes.size());
        for (long d = node->operation->dimensions - 1; d >= 0; d--) {
          if (d == node->operation->dimensions - 1) {
            acc_sizes[d] = 1;
            acc_sizes_pred[d] = 1;
          } else {
            acc_sizes_pred[d] = acc_sizes_pred[d + 1] * pred->shape[d + 1];
            acc_sizes[d] = acc_sizes[d + 1] * node->operation->shape[d + 1];
          }
        }
        // calculate start
        size_t start = 0;
        std::vector<long> step(node->operation->dimensions);
        for (long d = 0; d < step.size(); d++) {
          start += slice->start[d] * acc_sizes_pred[d];
        }
        index_defs += "index = " + to_string(start);
        // accumulate index
        for (long d = 0; d < node->operation->dimensions; d++) {
          index_defs +=
              " + (" +
              (d == 0 ? string("index")
                      : string("index %" + to_string(acc_sizes[d - 1]))) +
              ") / " + to_string(acc_sizes[d]) + " * " +
              to_string(slice->step[d] * (long)acc_sizes_pred[d]);
        }
        index_defs += ";\n";
        code = "index = old_index" + to_string(old_idx) + ";\n" + code;
        code = type + " " + name + " = v" + to_string(variable_index + 1) +
               ";\n" + code;
      } break;
      case FEXTEND: {
        FOperation *pred = node->predecessors[0]->operation;
        FExtend *extend = (FExtend *)node->operation->additional_data;
        unsigned int old_idx = num_indices++;
        index_defs += "int old_index" + to_string(old_idx) + " = index;\n";
        // flattened shape data
        std::vector<size_t> acc_sizes(node->operation->dimensions);
        std::vector<size_t> acc_sizes_pred(acc_sizes.size());
        for (long d = node->operation->dimensions - 1; d >= 0; d--) {
          if (d == node->operation->dimensions - 1) {
            acc_sizes[d] = 1;
            acc_sizes_pred[d] = 1;
          } else {
            acc_sizes_pred[d] = acc_sizes_pred[d + 1] * pred->shape[d + 1];
            acc_sizes[d] = acc_sizes[d + 1] * node->operation->shape[d + 1];
          }
        }
        // calculate start
        index_defs += "index = 0";
        std::string set_zero_cond = "if(";
        // accumulate index
        for (long d = 0; d < node->operation->dimensions; d++) {
          long step = extend->step[d];
          bool inv = step < 0;
          if (inv)
            step = -step;
          std::string dim_idx =
              "((" +
              (d == 0 ? string("index")
                      : string("index %" + to_string(acc_sizes[d - 1]))) +
              ") / " + to_string(acc_sizes[d]) + " - " +
              to_string(extend->start[d]) + ") / " + to_string(step);
          if (d != 0)
            set_zero_cond += " || ";
          // if di < start
          set_zero_cond +=
              "(" +
              (d == 0 ? string("index")
                      : string("index %" + to_string(acc_sizes[d - 1]))) +
              ") / " + to_string(acc_sizes[d]) + " < " +
              to_string(extend->start[d]);
          // if di % step != 0
          set_zero_cond +=
              " || ((" +
              (d == 0 ? string("index")
                      : string("index %" + to_string(acc_sizes[d - 1]))) +
              ") / " + to_string(acc_sizes[d]) + " - " +
              to_string(extend->start[d]) + ") % " + to_string(step) + " != 0";
          // if di >= shape
          set_zero_cond +=
              " || " + dim_idx + " >= " + to_string(pred->shape[d]);

          // finish index
          if (inv)
            dim_idx =
                "(" + to_string(pred->shape[d]) + " - " + dim_idx + " - 1)";
          index_defs += " + " + dim_idx + " * " + to_string(acc_sizes_pred[d]);
        }
        index_defs += ";\nif(index < 0) index = 0;\n";
        code = set_zero_cond + ") v" + to_string(variable_index) + " = 0;\n" +
               code;
        code = "index = old_index" + to_string(old_idx) + ";\n" + code;
        code = type + " " + name + " = v" + to_string(variable_index + 1) +
               ";\n" + code;
      } break;
      case FREPEAT: {
        const FOperation *op = node->operation;
        FOperation *pred = node->predecessors[0]->operation;
        unsigned int old_idx = num_indices++;
        index_defs += "int old_index" + to_string(old_idx) + " = index;\n";
        // add to index_defs a redefinition of index, so that we remap to src
        // data
        // calculate number of elements per dimension entry for destination and
        // source
        std::vector<size_t> acc_sizes_d(op->dimensions);
        std::vector<size_t> acc_sizes_s(op->dimensions);
        acc_sizes_d[op->dimensions - 1] = 1;
        acc_sizes_s[op->dimensions - 1] = 1;
        for (int dim = op->dimensions - 2; dim >= 0; dim--) {
          acc_sizes_d[dim] = acc_sizes_d[dim + 1] * op->shape[dim + 1];
          acc_sizes_s[dim] = acc_sizes_s[dim + 1] * pred->shape[dim + 1];
        }
        // to get the index in the source array we first calculate the indices
        // and reproject
        index_defs += "{\nint working_index = index;\nindex = 0;\n";
        for (int dim = 0; dim < op->dimensions; dim++) {
          index_defs += "index += ((working_index /" +
                        to_string(acc_sizes_d[dim]) + ") % " +
                        to_string(pred->shape[dim]) + ") * " +
                        to_string(acc_sizes_s[dim]) + ";\n";
          index_defs +=
              "working_index %= " + to_string(acc_sizes_d[dim]) + ";\n";
        }
        index_defs += "}\n";
        code = "index = old_index" + to_string(old_idx) + ";\n" + code;
        code = type + " " + name + " = v" + to_string(variable_index + 1) +
               ";\n" + code;
      } break;
      case FTRANSPOSE: {
        const FOperation *op = node->operation;
        const int *transposition = (int *)op->additional_data;
        FOperation *pred = node->predecessors[0]->operation;
        unsigned int old_idx = num_indices++;
        index_defs += "int old_index" + to_string(old_idx) + " = index;\n";
        // add to index_defs a redefinition of index, so that we remap to src
        // data
        // calculate number of elements per dimension entry for destination and
        // source
        std::vector<size_t> acc_sizes_d(op->dimensions);
        std::vector<size_t> acc_sizes_s(op->dimensions);
        acc_sizes_d[op->dimensions - 1] = 1;
        acc_sizes_s[op->dimensions - 1] = 1;
        for (int dim = op->dimensions - 2; dim >= 0; dim--) {
          acc_sizes_d[dim] = acc_sizes_d[dim + 1] * op->shape[dim + 1];
          acc_sizes_s[dim] = acc_sizes_s[dim + 1] * pred->shape[dim + 1];
        }
        // to get the index in the source array we first calculate the indices
        // and reproject
        index_defs += "{\nint working_index = index;\nindex = 0;\n";
        for (int dim = 0; dim < op->dimensions; dim++) {
          index_defs += "index += (working_index /" +
                        to_string(acc_sizes_d[dim]) + ") * " +
                        to_string(acc_sizes_s[transposition[dim]]) + ";\n";
          index_defs +=
              "working_index %= " + to_string(acc_sizes_d[dim]) + ";\n";
        }
        index_defs += "}\n";
        code = "index = old_index" + to_string(old_idx) + ";\n" + code;
        code = type + " " + name + " = v" + to_string(variable_index + 1) +
               ";\n" + code;
      } break;
      default:
        break;
      }
    // insert our indexing logic into the queue after the children
    if (!index_defs.empty())
      todo.push_front({nullptr, index_defs});
    // push predecessors dfs
    if (push_pred)
      for (int i = 0; i < node->num_predecessor; i++)
        todo.push_front(
            {node->predecessors[i], "v" + to_string(++variable_index)});
  }
  code = "int index = get_global_id(0);\n" + code;
  return code;
}
static std::string generateEagerCode(FGraphNode *node) {
  using namespace std;
  string code = "__kernel void execute_graph(__global " +
                typeString(node->operation->data_type) + "* R";
  // generate parameters
  switch (node->operation->op_type) {
  case FSTORE:
  case FCONST:
  case FLATTEN:
  case FRESHAPE:
  case FNUM_OPERATION_TYPES:
    break; // should not happen
  case FMATMUL:
    code += ", long num_entriesR, long l, long m, long n";
    for (int i = 0; i < 2; i++) {
      code += ", __constant " +
              typeString(node->predecessors[i]->operation->data_type) + "* P" +
              to_string(i) + ", long num_entries" + to_string(i) +
              ", int dimensions" + to_string(i);
    }
    break;
  case FREDUCE_SUM:
  case FREDUCE_MUL:
  case FSLICE:
  case FREPEAT:
  case FTRANSPOSE:
  case FEXTEND:
    // TODO
  default:
    for (int i = 0; i < node->num_predecessor; i++)
      code += ", __constant " +
              typeString(node->predecessors[i]->operation->data_type) + "* P" +
              to_string(i) + ", long num_entries" + to_string(i);
    break;
  }
  code += "){\nconst int index = get_global_id(0);\n";
  // generate code
  switch (node->operation->op_type) {
  case FADD:
    code += "if(index >= num_entries0 && index >= num_entries1) "
            "return;\nR[index] = "
            "P0[index%num_entries0] + P1[index%num_entries1];";
    break;
  case FSUB:
    code += "if(index >= num_entries0 && index >= num_entries1) "
            "return;\nR[index] = "
            "P0[index%num_entries0] - P1[index%num_entries1];";
    break;
  case FMUL:
    code += "if(index >= num_entries0 && index >= num_entries1) "
            "return;\nR[index] = "
            "P0[index%num_entries0] * P1[index%num_entries1];";
    break;
  case FDIV:
    code += "if(index >= num_entries0 && index >= num_entries1) "
            "return;\nR[index] = "
            "P0[index%num_entries0] / P1[index%num_entries1];";
    break;
  case FPOW: {
    code += "if(index >= num_entries0 && index >= num_entries1) return;\n";
    FOperation *x = node->predecessors[0]->operation;
    FOperation *y = node->predecessors[1]->operation;
    string type = typeString(node->operation->data_type);
    if ((x->data_type == F_FLOAT32 || x->data_type == F_FLOAT64) &&
        (y->data_type == F_FLOAT32 || y->data_type == F_FLOAT64))
      code += "R[index] = pow((" + type + ")P0[index%num_entries0], (" + type +
              ")P1[index%num_entries1]);";
    else if (x->data_type == F_INT64 &&
             (y->data_type == F_INT32 || y->data_type == F_INT64))
      code += "R[index] "
              "= (long)pown((double)P0[index%num_entries0], "
              "(int)P1[index%num_entries1]);";
    else if (x->data_type == F_INT32 &&
             (y->data_type == F_INT32 || y->data_type == F_INT64))
      code += "R[index] = "
              "(int)pown((float)P0[index%num_entries0], "
              "(int)P1[index%num_entries1]);";
    else
      code += "R[index] = "
              "pow((double)P0[index%num_entries0], "
              "(double)P1[index%num_entries1]);";
  } break;
  case FNEG:
    code += "if(index >= num_entries0) return;\n";
    code += "R[index] = "
            "-P0[index];";
    break;
  case FLOG:
    code += "if(index >= num_entries0) return;\n";
    code += "R[index] = "
            "log(P0[index]);";
    break;
  case FSIGN:
    code += "if(index >= num_entries0) return;\n";
    code += "R[index] = "
            "P0[index] >= 0 ? 1 : -1;";
    break;
  case FEVEN:
    code += "if(index >= num_entries0) return;\n";
    code += "R[index] = "
            "P0[index] % 2 == 0 ? 1 : 0;";
    break;
  case FMATMUL: {
    code += "if(index >= num_entriesR) return;\n";
    code += typeString(node->operation->data_type) + " res = 0;\n";
    code += "long j = (index % (l * n)) / n;\n";
    code += "long k = (index % (l * n)) % n;\n";
    code +=
        "long base_p0 = dimensions0 > 2 ? (index / (l * n)) * (l * m) : 0;\n";
    code +=
        "long base_p1 = dimensions1 > 2 ? (index / (l * n)) * (m * n) : 0;\n";
    code += "for(int i = 0; i < m; i++){\n res += P0[base_p0 + j * m + i] * "
            "P1[base_p1 + i * n + k];\n}";
    code += "R[index] = res;\n";
    break;
  }
  case FABS:
    code += "if(index >= num_entries0) return;\n";
    code += "R[index] = "
            "P0[index] < 0 ? -P0[index] : P0[index];";
    break;
  case FLOG2:
    code += "if(index >= num_entries0) return;\n";
    code += "R[index] = "
            "log2(P0[index]);";
    break;
  case FLOG10:
    code += "if(index >= num_entries0) return;\n";
    code += "R[index] = "
            "log10(P0[index]);";
    break;
  // case FLATTEN:
  // case FRESHAPE:
  case FCONVERSION:
    code += "if(index >= num_entries0) return;\n";
    code +=
        "R[index] = (" + typeString(node->operation->data_type) + ")P0[index];";
    break;
  case FMIN:
    code += "if(index >= num_entries0 && index >= num_entries1) return;\n";
    code += typeString(node->predecessors[0]->operation->data_type) +
            " a = P0[index%num_entries0];\n";
    code += typeString(node->predecessors[1]->operation->data_type) +
            " b = P1[index%num_entries1];\n";
    code += "R[index] = a < b ? a : b";
    break;
  case FMAX:
    code += "if(index >= num_entries0 && index >= num_entries1) return;\n";
    code += typeString(node->predecessors[0]->operation->data_type) +
            " a = P0[index%num_entries0];\n";
    code += typeString(node->predecessors[1]->operation->data_type) +
            " b = P1[index%num_entries1];\n";
    code += "R[index] = a > b ? a : b";
    break;
  case FLESS:
    code += "if(index >= num_entries0 && index >= num_entries1) return;\n";
    code += typeString(node->predecessors[0]->operation->data_type) +
            " a = P0[index%num_entries0];\n";
    code += typeString(node->predecessors[1]->operation->data_type) +
            " b = P1[index%num_entries1];\n";
    code += "R[index] = a < b ? 1 : 0";
    break;
  case FEQUAL:
    code += "if(index >= num_entries0 && index >= num_entries1) return;\n";
    code += typeString(node->predecessors[0]->operation->data_type) +
            " a = P0[index%num_entries0];\n";
    code += typeString(node->predecessors[1]->operation->data_type) +
            " b = P1[index%num_entries1];\n";
    code += "R[index] = a == b ? 1 : 0";
    break;
  case FGREATER:
    code += "if(index >= num_entries0 && index >= num_entries1) return;\n";
    code += typeString(node->predecessors[0]->operation->data_type) +
            " a = P0[index%num_entries0];\n";
    code += typeString(node->predecessors[1]->operation->data_type) +
            " b = P1[index%num_entries1];\n";
    code += "R[index] = a > b ? 1 : 0";
    break;
  case FREDUCE_SUM:
  case FREDUCE_MUL:
  case FSLICE:
  case FREPEAT:
  case FTRANSPOSE:
  case FEXTEND:
  case FNUM_OPERATION_TYPES:
    break;
  }
  code += "\n}";
  flogging(F_DEBUG, std::string("Eager Kernel Generation for ") +
                        fop_to_string[node->operation->op_type] + ": " + code);
  return code;
}
#endif
