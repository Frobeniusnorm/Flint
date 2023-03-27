/* Copyright 2022 David Schwarzbeck

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef GRADIENTS_CPP
#define GRADIENTS_CPP
#include "../flint.h"
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
  if (adjoint->operation->dimensions > node->operation->dimensions) {
    size_t diff = adjoint->operation->dimensions - node->operation->dimensions;
    FGraphNode *res = adjoint;
    for (int i = 0; i < diff; i++) {
      res = freduce_sum(res, 0);
    }
    return res;
  } else if (adjoint->operation->dimensions < node->operation->dimensions) {
    size_t diff = node->operation->dimensions - adjoint->operation->dimensions;
    std::vector<size_t> new_shape(node->operation->dimensions);
    std::vector<int> repetitions(node->operation->dimensions, 0);
    for (int i = 0; i < diff; i++) {
      new_shape[i] = 1;
      repetitions[i] = node->operation->shape[i] - 1;
    }
    for (int i = diff; i < new_shape.size(); i++)
      new_shape[i] = adjoint->operation->shape[i - diff];
    FGraphNode *res = freshape(adjoint, new_shape.data(), new_shape.size());
    res = frepeat(res, repetitions.data());
    return res;
  }
  return adjoint;
}
static std::string printShape(size_t *shape, int dim) {
  std::vector<size_t> sh(shape, shape + dim);
  return vectorString(sh);
}
template <typename T> static std::string printNode(FGraphNode *node) {
  std::string s = "";
  if (!node->result_data) {
    fExecuteGraph(node);
  }
  for (int i = 0; i < node->result_data->num_entries; i++)
    s += std::to_string(((T *)node->result_data->data)[i]) +
         (i == node->result_data->num_entries - 1 ? std::string("")
                                                  : std::string(", "));
  return s;
}
static FGraphNode *local_gradient(FGraphNode *y, FGraphNode *dx,
                                  FGraphNode *prev_adj) {
  switch (y->operation->op_type) {
  case FADD:
    return (dx == y->predecessors[0] || dx == y->predecessors[1]) ? prev_adj
                                                                  : nullptr;
  case FSUB: {
    if (dx == y->predecessors[0])
      return prev_adj;
    else if (dx == y->predecessors[1])
      return fneg(prev_adj);
    else
      return nullptr;
  }
  case FMUL: {
    if (y->predecessors[0] == dx) {
      return fmul(prev_adj, y->predecessors[1]);
    } else if (y->predecessors[1] == dx) {
      return fmul(prev_adj, y->predecessors[0]);
    } else
      return nullptr;
  }
  case FDIV: {
    FGraphNode *a = y->predecessors[0];
    FGraphNode *b = y->predecessors[1];
    if (a == dx) {
      // d(a / b)/da = d(a * b^(-1))/da = b^(-1)
      return fdiv(prev_adj, b);
    } else if (b == dx) {
      // d(a / b)/db = d(a * b^(-1))/db = -a * b^(-2)
      return fneg(fdiv(fmul(prev_adj, a), fpow(b, 2.)));
    } else
      return nullptr;
  }
  case FMATMUL: {
    FGraphNode *a = y->predecessors[0];
    FGraphNode *b = y->predecessors[1];
    if (a == dx) {
      std::vector<int> perm(b->operation->dimensions);
      for (int i = 0; i < perm.size() - 2; i++)
        perm[i] = i;
      perm[perm.size() - 2] = perm.size() - 1;
      perm[perm.size() - 1] = perm.size() - 2;
      return fmatmul(prev_adj, ftranspose(b, perm.data()));
    } else if (b == dx) {
      std::vector<int> perm(a->operation->dimensions);
      for (int i = 0; i < perm.size() - 2; i++)
        perm[i] = i;
      perm[perm.size() - 2] = perm.size() - 1;
      perm[perm.size() - 1] = perm.size() - 2;
      return fmatmul(ftranspose(a, perm.data()), prev_adj);
    } else {
      return nullptr;
    }
  }
  case FPOW: {
    FGraphNode *a = y->predecessors[0];
    FGraphNode *b = y->predecessors[1];
    if (a == dx) {
      // x^b / dx = b*x^(b-1)
      return fmul(prev_adj, fmul(b, fpow(a, fsub(b, 1))));
    } else if (b == dx) {
      // a^x / dx = a^x * ln(a)
      // has to be zero when a < 0 since not differentiable
      return fmul(prev_adj, fmul(fmul(fadd(fsign(a), 1), 0.5),
                                 fmul(fpow(a, b), flog(fabs_g(a)))));
    } else
      return nullptr;
  }
  case FNEG: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx) {
      return fneg(prev_adj);
    } else
      return nullptr;
  }
  case FLOG: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx)
      return fdiv(prev_adj, a);
    else
      return nullptr;
  }
  case FLOG2: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx)
      return fdiv(prev_adj, fmul(a, log(2.0)));
    else
      return nullptr;
  }
  case FLOG10: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx)
      return fdiv(prev_adj, fmul(a, log(10.0)));
    else
      return nullptr;
  }
  case FRESHAPE:
  case FLATTEN: {
    // reproject adjacent into previous shape, should be okay since
    // shape(prev_adj) = shape(y)
    FGraphNode *prev = y->predecessors[0];
    return freshape(prev_adj, prev->operation->shape,
                    prev->operation->dimensions);
  }
  case FCONVERSION:
    return prev_adj;
  case FMIN: {
    FGraphNode *a = y->predecessors[0];
    FGraphNode *b = y->predecessors[1];
    if (a == dx) {
      return fmul(prev_adj, fadd(fless(a, b), fequal(a, b)));
    } else if (b == dx)
      return fmul(prev_adj, fgreater(a, b));
    else
      return nullptr;
  }
  case FMAX: {
    FGraphNode *a = y->predecessors[0];
    FGraphNode *b = y->predecessors[1];
    if (a == dx)
      return fmul(prev_adj, fadd(fgreater(a, b), fequal(a, b)));
    else if (b == dx)
      return fmul(prev_adj, fless(a, b));
    else
      return nullptr;
  }
  case FABS: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx) {
      return fmul(prev_adj, fsub(fsign(a), fequal(a, 0.0)));
    } else
      return nullptr;
  }
  case FREDUCE_SUM: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx) {
      FOperation *op = y->operation;
      const int dim = ((int *)op->additional_data)[0];
      std::vector<int> rep(a->operation->dimensions);
      std::vector<size_t> ns(a->operation->dimensions);
      for (int i = 0; i < rep.size(); i++) {
        rep[i] = i == dim ? a->operation->shape[i] - 1 : 0;
        ns[i] = i != dim ? a->operation->shape[i] : 1;
      }
      return frepeat(freshape(prev_adj, ns.data(), ns.size()), rep.data());
    } else
      return nullptr;
  }
  case FREDUCE_MUL: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx) {
      FOperation *op = y->operation;
      const int dim = ((int *)op->additional_data)[0];
      std::vector<int> rep(a->operation->dimensions);
      std::vector<size_t> ns(a->operation->dimensions);
      for (int i = 0; i < rep.size(); i++) {
        rep[i] = i == dim ? a->operation->shape[i] - 1 : 0;
        ns[i] = i != dim ? a->operation->shape[i] : 1;
      }
      FGraphNode *zero_node = fequal(a, 0.0);
      // the normal gradient would be y/a, this does not work for a_i = 0, but
      // at first we calculate the gradient for every a_i != 0 broadcast y
      FGraphNode *ls = frepeat(freshape(y, ns.data(), ns.size()), rep.data());
      // calculate y/a and remove division by 0 case (it does not matter what we
      // add in that case, since we multiply by 1 - fequal(a, 0.0), just avoid /
      // 0 for portability)
      FGraphNode *lg = fmul(
          fsub_ici(1, zero_node),
          fdiv(ls, fadd(a, zero_node))); // explicitly removing divison by 0
      // to compute a_i = 0 case we set each 0-entry to 1 and repeat the
      // computation, this yields the correct gradients only for the entries
      // where a_i = 0
      FGraphNode *zg =
          fmul(zero_node, frepeat(freshape(freduce_mul(fadd(a, zero_node), dim),
                                           ns.data(), ns.size()),
                                  rep.data()));
      // now we can add both gradients and multiply with the previous adjoint
      return fmul(frepeat(freshape(prev_adj, ns.data(), ns.size()), rep.data()),
                  fadd(lg, zg));
    } else
      return nullptr;
  }
  case FREPEAT: {
    FGraphNode *a = y->predecessors[0];
    std::vector<long> start(y->operation->dimensions, 0);
    std::vector<long> end(y->operation->dimensions);
    long rep_mul = 1;
    for (int i = 0; i < start.size(); i++) {
      if (y->operation->shape[i] != a->operation->shape[i])
        end[i] = y->operation->shape[i] / a->operation->shape[i];
      else
        end[i] = y->operation->shape[i];
      rep_mul *= (y->operation->shape[i] / a->operation->shape[i]);
    }
    return fmul(fslice(prev_adj, start.data(), end.data()), rep_mul);
  }
  case FTRANSPOSE: {
    int *transp = ((int *)y->operation->additional_data);
    return ftranspose(prev_adj, transp);
  }
  case FSLICE: {
    const FGraphNode *a = y->predecessors[0];
    const FSlice *slice = (FSlice *)y->operation->additional_data;
    std::vector<size_t> start(a->operation->dimensions);
    for (int i = 0; i < a->operation->dimensions; i++) {
      start[i] = slice->step[i] >= 0 ? slice->start[i] : slice->end[i] + 1;
    }
    return fextend_step(prev_adj, a->operation->shape, start.data(),
                        slice->step);
  }
  case FEXTEND: {
    const FGraphNode *a = y->predecessors[0];
    const FExtend *extend = (FExtend *)y->operation->additional_data;
    std::vector<long> start(a->operation->dimensions);
    std::vector<long> ends(a->operation->dimensions);
    std::vector<long> steps(a->operation->dimensions);
    for (int i = 0; i < a->operation->dimensions; i++) {
      start[i] = extend->start[i];
      ends[i] = a->operation->shape[i] * extend->step[i] + extend->start[i];
      steps[i] = extend->step[i];
    }
    return fslice_step(prev_adj, start.data(), ends.data(), steps.data());
  }
  case FSIGN:
  case FEVEN:
  case FLESS:
  case FEQUAL:
  case FGREATER:
    return constant_tensor(0.0, F_FLOAT64, y->operation->shape,
                           y->operation->dimensions);
  default:
    return nullptr;
  }
}

FGraphNode *fCalculateGradient(FGraphNode *y, const FGraphNode *dx) {
  using namespace std;
  // to store gradients per node
  unordered_map<FGraphNode *, FGraphNode *> adjoints;
  // fixpoint iteration
  list<FGraphNode *> working;
  unordered_set<FGraphNode *> in_working;
  working.push_back(y);
  in_working.insert(y);

  // initialize
  adjoints[y] = constant_tensor(1., F_FLOAT64, y->operation->shape,
                                y->operation->dimensions);
  FGraphNode *sol = nullptr;
  while (!working.empty()) {
    FGraphNode *curr = working.front();
    working.pop_front();
    in_working.erase(curr);
    if (curr == dx) {
      sol = adjoints[curr];
    }
    if (curr->operation->op_type == FSTORE)
      continue;
    FGraphNode *adj = adjoints[curr];
    for (int i = 0; i < curr->num_predecessor; i++) {
      FGraphNode *parent = curr->predecessors[i];
      FGraphNode *local_grad = local_gradient(curr, parent, adj);
      if (adjoints.contains(parent)) {
        adjoints[parent] =
            fadd(adjoints[parent], unbroadcast(local_grad, parent));
      } else {
        adjoints.insert({parent, unbroadcast(local_grad, parent)});
      }
      if (!in_working.contains(parent)) {
        working.push_back(parent);
        in_working.insert(parent);
      }
    }
  }
  if (!sol)
    flogging(F_WARNING, "Operation graph did not contain the derivative!");
  if (sol->operation->data_type != F_FLOAT64)
    flogging(F_ERROR, "I did something wrong!"); // TODO remove this
  return sol;
}
#endif
