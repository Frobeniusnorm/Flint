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

  This file includes the implementation of the gradient calculation functions
*/

#ifndef GRADIENTS_CPP
#define GRADIENTS_CPP
#include "../flint.h"
#include "backend_ocl/comp.hpp"
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
/**
 * Slides kernel along the shape of a and accumulates for each element of a the
 * values of the kernel that are slid against it. Additionally reprojects the
 * values of the adjoint gradient of the convolution operation to the position
 * where each value was calculated in a and multiplies it with the corresponding
 * elements of the kernel before they are accumulated. Finally this yield the
 * gradient of a.
 */
static FGraphNode *gradient_convolve(FGraphNode *a, FGraphNode *kernel,
                                     FGraphNode *prev_adj,
                                     const unsigned int *steps) {
  if (!kernel->result_data)
    fExecuteGraph(kernel);
  if (!prev_adj->result_data)
    fExecuteGraph(prev_adj);
  FGraphNode *gradient = new FGraphNode();
  gradient->num_predecessor = 2;
  gradient->predecessors = safe_mal<FGraphNode *>(2);
  gradient->predecessors[0] = kernel;
  gradient->predecessors[1] = prev_adj;
  kernel->reference_counter++;
  prev_adj->reference_counter++;
  gradient->result_data = nullptr;
  gradient->reference_counter = 0;
  FOperation op;
  op.data_type = F_FLOAT64;
  op.dimensions = a->operation.dimensions;
  op.shape = safe_mal<size_t>(op.dimensions);
  memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
  op.op_type = FGRADIENT_CONVOLVE;
  op.additional_data = safe_mal<unsigned int>(a->operation.dimensions - 1);
  memcpy(op.additional_data, steps,
         (a->operation.dimensions - 1) * sizeof(unsigned int));
  gradient->operation = op;
  configureGradientInformation(gradient, {kernel, prev_adj});
  return gradient;
}
static FGraphNode *unbroadcast(FGraphNode *adjoint, const FGraphNode *node) {
  if (adjoint->operation.dimensions > node->operation.dimensions) {
    size_t diff = adjoint->operation.dimensions - node->operation.dimensions;
    FGraphNode *res = adjoint;
    for (int i = 0; i < diff; i++) {
      res = freduce_sum(res, 0);
    }
    return res;
  } else if (adjoint->operation.dimensions < node->operation.dimensions) {
    size_t diff = node->operation.dimensions - adjoint->operation.dimensions;
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
static std::string printShape(size_t *shape, int dim) {
  std::vector<size_t> sh(shape, shape + dim);
  return vectorString(sh);
}
template <typename T>
static std::string printNode(FGraphNode *node, int dim, int *b) {
  std::string prev = "";
  for (int i = 0; i < dim; i++)
    prev += " ";
  std::string s = "[";
  if (dim == node->operation.dimensions - 1) {
    for (int i = 0; i < node->operation.shape[dim]; i++) {
      s += std::to_string(((T *)node->result_data->data)[*b + i]);
      if (i != node->operation.shape[dim] - 1)
        s += ", ";
    }
    (*b) += node->operation.shape[dim];
  } else {
    for (int i = 0; i < node->operation.shape[dim]; i++)
      s += printNode<T>(node, dim + 1, b) + ",\n" + prev;
    s = s.substr(0, s.size() - 2 - prev.size());
  }
  return s + "]";
}
template <typename T> static std::string printNode(FGraphNode *node) {
  if (!node->result_data) {
    fExecuteGraph(node);
  }
  int b = 0;
  return printNode<T>(node, 0, &b);
}
static FGraphNode *local_gradient(FGraphNode *y, int dx_i,
                                  FGraphNode *prev_adj) {
  FGraphNode *dx = y->predecessors[dx_i];
  switch (y->operation.op_type) {
  case FADD:
    return (dx_i == 0 || dx_i == 1) ? prev_adj : nullptr;
  case FSUB: {
    if (dx_i == 0)
      return prev_adj;
    else if (dx_i == 1)
      return fneg(prev_adj);
    else
      return nullptr;
  }
  case FMUL: {
    if (0 == dx_i) {
      return fmul(prev_adj, y->predecessors[1]);
    } else if (1 == dx_i) {
      return fmul(prev_adj, y->predecessors[0]);
    } else
      return nullptr;
  }
  case FDIV: {
    FGraphNode *a = y->predecessors[0];
    FGraphNode *b = y->predecessors[1];
    if (0 == dx_i) {
      // d(a / b)/da = d(a * b^(-1))/da = b^(-1)
      return fdiv(prev_adj, b);
    } else if (1 == dx_i) {
      // d(a / b)/db = d(a * b^(-1))/db = -a * b^(-2)
      return fneg(fdiv(fmul(prev_adj, a), fpow(b, 2.)));
    } else
      return nullptr;
  }
  case FMATMUL: {
    FGraphNode *a = y->predecessors[0];
    FGraphNode *b = y->predecessors[1];
    if (0 == dx_i) {
      std::vector<int> perm(b->operation.dimensions);
      for (int i = 0; i < perm.size() - 2; i++)
        perm[i] = i;
      perm[perm.size() - 2] = perm.size() - 1;
      perm[perm.size() - 1] = perm.size() - 2;
      return fmatmul(prev_adj, ftranspose(b, perm.data()));
    } else if (1 == dx_i) {
      std::vector<int> perm(a->operation.dimensions);
      for (int i = 0; i < perm.size() - 2; i++)
        perm[i] = i;
      perm[perm.size() - 2] = perm.size() - 1;
      perm[perm.size() - 1] = perm.size() - 2;
      return fmatmul(ftranspose(a, perm.data()), prev_adj);
    } else {
      return nullptr;
    }
  }
  case FCONCAT: {
    FGraphNode *a = y->predecessors[0];
    FGraphNode *b = y->predecessors[1];
    unsigned int ax = *((unsigned int *)y->operation.additional_data);
    if (0 == dx_i) {
      std::vector<long> start(a->operation.dimensions);
      std::vector<long> stop(a->operation.dimensions);
      for (int i = 0; i < a->operation.dimensions; i++) {
        start[i] = 0;
        stop[i] = a->operation.shape[i];
      }
      return fslice(prev_adj, start.data(), stop.data());
    } else if (1 == dx_i) {
      std::vector<long> start(b->operation.dimensions);
      std::vector<long> stop(b->operation.dimensions);
      for (int i = 0; i < b->operation.dimensions; i++) {
        start[i] = ax == i ? a->operation.shape[i] : 0;
        stop[i] = ax == i ? a->operation.shape[i] + b->operation.shape[i]
                          : b->operation.shape[i];
      }
      return fslice(prev_adj, start.data(), stop.data());
    } else
      return nullptr;
  }
  case FINDEX: {
    FGraphNode *a = y->predecessors[0];
    FGraphNode *b = y->predecessors[1];
    if (0 == dx_i) {
      FGraphNode *g =
          fconstant_d(0, a->operation.shape, a->operation.dimensions);
      return findex_set(g, prev_adj, b);
    } else
      return fconstant_d(0, b->operation.shape, b->operation.dimensions);
  } break;
  case FSET_INDEX: {
    FGraphNode *b = y->predecessors[1];
    FGraphNode *i = y->predecessors[2];
    // a[i] = b
    if (0 == dx_i) {
      FGraphNode *g =
          fconstant_d(0, b->operation.shape, b->operation.dimensions);
      // remove values that have been overwritten
      return findex_set(prev_adj, g, i);
    } else {
      // filter for b relevant elements
      return findex(prev_adj, i);
    }
  } break;
  case FSLIDING_WINDOW: {
    FGraphNode *a = y->predecessors[0];
    if (0 == dx_i) {
      // TODO while this works in reasonable time it has a grotesque memory
      // consumption Better would be a custom function for the gradient
      FSlidingWindow *sliding_win =
          (FSlidingWindow *)y->operation.additional_data;
      FGraphNode *res =
          funslide_window(prev_adj, a->operation.shape, sliding_win->step);
      return res;
    } else
      return nullptr;
  } break;
  case FSLIDE:
  case FCONVOLVE: {
    FGraphNode *a = y->predecessors[0];
    FGraphNode *kernel = y->predecessors[1];
    const unsigned int *steps = (unsigned int *)y->operation.additional_data;
    if (0 == dx_i) {
      return gradient_convolve(a, kernel, prev_adj, steps);
    } else if (1 == dx_i) {
      if (y->operation.op_type == FCONVOLVE) {
        unsigned int *steps = (unsigned int *)y->operation.additional_data;
        std::vector<unsigned int> new_steps(kernel->operation.dimensions);
        memcpy(new_steps.data(), steps,
               (kernel->operation.dimensions - 1) * sizeof(unsigned int));
        new_steps[kernel->operation.dimensions - 1] =
            kernel->operation.shape[kernel->operation.dimensions - 1];
        FGraphNode *na =
            fsliding_window(a, kernel->operation.shape, new_steps.data());
        // broadcast along first dimension
        prev_adj = fflatten(prev_adj);
        // repeat to fit correct shape
        for (int i = 1; i < na->operation.dimensions; i++)
          prev_adj = fexpand(prev_adj, i, na->operation.shape[i]);
        na = fmul(na, prev_adj);
        FGraphNode *res = nullptr;
        if (flintInitializedBackends() & FLINT_BACKEND_ONLY_GPU) {
          // we check if we can subdivide the reduction task for better parallel
          // distribution
          int subdivs[] = {128, 100, 50, 10, 7, 5, 4, 3, 2};
          for (int i = 0; i < sizeof(subdivs) / sizeof(int); i++) {
            if (na->operation.shape[0] % subdivs[i] == 0 &&
                na->operation.shape[0] / subdivs[i] > 1) {
              std::vector<size_t> subdiv_shape(na->operation.dimensions + 1);
              for (int i = 0; i < na->operation.dimensions; i++)
                subdiv_shape[i + 1] = na->operation.shape[i];
              subdiv_shape[0] = subdivs[i];
              subdiv_shape[1] /= subdivs[i];
              res = freduce_sum(
                  freshape(na, subdiv_shape.data(), subdiv_shape.size()), 1);
              break;
            }
          }
          if (!res)
            res = na;
        } else
          res = na;
        res = freduce_sum(res, 0);
        return res;
      } else
        return fslide(a, prev_adj,
                      (unsigned int *)y->operation.additional_data);
    }
    return nullptr;
  }
  case FGRADIENT_CONVOLVE: {
    // for the derivation of a derivation
    FGraphNode *kernel = y->predecessors[0];
    FGraphNode *a = y->predecessors[1];
    if (1 == dx_i) {
      const unsigned int *steps = (unsigned int *)y->operation.additional_data;
      // fuck this noise, i am writing a custom function i cant take this
      // anymore
      if (!kernel->result_data)
        fExecuteGraph(kernel);
      FGraphNode *gradient = new FGraphNode();
      gradient->num_predecessor = 2;
      gradient->predecessors = safe_mal<FGraphNode *>(2);
      gradient->predecessors[0] = kernel;
      gradient->predecessors[1] = prev_adj;
      kernel->reference_counter++;
      prev_adj->reference_counter++;
      gradient->result_data = nullptr;
      gradient->reference_counter = 0;
      FOperation op;
      op.data_type = F_FLOAT64;
      op.dimensions = a->operation.dimensions;
      op.shape = safe_mal<size_t>(op.dimensions);
      memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
      op.op_type = FGRADIENT_CONVOLVE;
      op.additional_data = safe_mal<unsigned int>(a->operation.dimensions - 1);
      memcpy(op.additional_data, steps,
             (a->operation.dimensions - 1) * sizeof(unsigned int));
      gradient->operation = op;
      configureGradientInformation(gradient, {kernel, prev_adj});
      return gradient;
    } else if (0 == dx_i) {
      return fslide(prev_adj, a, (unsigned int *)y->operation.additional_data);
    }
    return nullptr;
  } break;
  case FPOW: {
    FGraphNode *a = y->predecessors[0];
    FGraphNode *b = y->predecessors[1];
    if (0 == dx_i) {
      // x^b / dx = b*x^(b-1)
      return fmul(prev_adj, fmul(b, fpow(a, fsub(b, 1))));
    } else if (1 == dx_i) {
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
    return freshape(prev_adj, prev->operation.shape,
                    prev->operation.dimensions);
  }
  case FCONVERSION:
    return prev_adj;
  case FMIN: {
    FGraphNode *a = y->predecessors[0];
    FGraphNode *b = y->predecessors[1];
    if (0 == dx_i) {
      return fmul(prev_adj, fadd(fless(a, b), fequal(a, b)));
    } else if (1 == dx_i)
      return fmul(prev_adj, fgreater(a, b));
    else
      return nullptr;
  }
  case FMAX: {
    FGraphNode *a = y->predecessors[0];
    FGraphNode *b = y->predecessors[1];
    if (0 == dx_i)
      return fmul(prev_adj, fadd(fgreater(a, b), fequal(a, b)));
    else if (1 == dx_i)
      return fmul(prev_adj, fadd(fless(a, b), fequal(a, b)));
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
  case FSQRT: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx) {
      return fdiv(prev_adj, fmul_ci(y, 2));
    } else
      return nullptr;
  }
  case FEXP: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx) {
      return fmul(prev_adj, y);
    } else
      return nullptr;
  }
  case FSIN: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx) {
      return fmul(prev_adj, fcos(a));
    } else
      return nullptr;
  }
  case FCOS: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx) {
      return fmul(prev_adj, fneg(fsin(a)));
    } else
      return nullptr;
  }
  case FTAN: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx) {
      return fmul(prev_adj, fpow(fcos(a), -2));
    } else
      return nullptr;
  }
  case FASIN: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx) {
      return fdiv(prev_adj, fsqrt_g(fsub_icd(1.0, fmul(a, a))));
    } else
      return nullptr;
  }
  case FACOS: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx) {
      return fdiv(prev_adj, fneg(fsqrt_g(fsub_icd(1.0, fmul(a, a)))));
    } else
      return nullptr;
  }
  case FATAN: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx) {
      return fdiv(prev_adj, fadd_ci(fmul(a, a), 1));
    } else
      return nullptr;
  }
  case FREDUCE_MAX:
  case FREDUCE_MIN: {
    FGraphNode *a = y->predecessors[0];
    unsigned int ax = ((unsigned int *)y->operation.additional_data)[0];
    // work with extend to readjust the node to the same shape as before by
    // repetition, then compare it with equal and multiply the 0-1 tensor with
    // previous adjoint.
    FGraphNode *n = fequal(a, fexpand(y, ax, a->operation.shape[ax]));
    return fmul(fexpand(prev_adj, ax, a->operation.shape[ax]), n);
  }
  case FREDUCE_SUM: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx) {
      const FOperation op = y->operation;
      const int dim = ((int *)op.additional_data)[0];
      std::vector<int> rep(a->operation.dimensions);
      std::vector<size_t> ns(a->operation.dimensions);
      for (int i = 0; i < rep.size(); i++) {
        rep[i] = i == dim ? a->operation.shape[i] - 1 : 0;
        ns[i] = i != dim ? a->operation.shape[i] : 1;
      }
      return frepeat(freshape(prev_adj, ns.data(), ns.size()), rep.data());
    } else
      return nullptr;
  }
  case FREDUCE_MUL: {
    FGraphNode *a = y->predecessors[0];
    if (a == dx) {
      const FOperation op = y->operation;
      const int dim = ((int *)op.additional_data)[0];
      std::vector<int> rep(a->operation.dimensions);
      std::vector<size_t> ns(a->operation.dimensions);
      for (int i = 0; i < rep.size(); i++) {
        rep[i] = i == dim ? a->operation.shape[i] - 1 : 0;
        ns[i] = i != dim ? a->operation.shape[i] : 1;
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
    FGraphNode *grad = prev_adj;
    std::vector<size_t> orig_shape(prev_adj->operation.shape,
                                   prev_adj->operation.shape +
                                       prev_adj->operation.dimensions);
    for (int i = 0; i < a->operation.dimensions; i++) {
      if (a->operation.shape[i] != y->operation.shape[i]) {
        // reduce repeated gradient into correct shape
        std::vector<size_t> new_shape(orig_shape.size() + 1);
        // add extra dimension for reducing
        if (i > 0)
          std::memcpy(new_shape.data(), grad->operation.shape,
                      sizeof(size_t) * i);
        new_shape[i] = orig_shape[i] / a->operation.shape[i];
        new_shape[i + 1] = a->operation.shape[i];
        if (orig_shape.size() - i > 1)
          std::memcpy(new_shape.data() + i + 2, grad->operation.shape + i + 1,
                      sizeof(size_t) * (orig_shape.size() - i - 1));
        // reduce along that axis
        grad =
            freduce_sum(freshape(grad, new_shape.data(), new_shape.size()), i);
      }
    }
    return grad;
  }
  case FTRANSPOSE: {
    int *transp = ((int *)y->operation.additional_data);
    return ftranspose(prev_adj, transp);
  }
  case FSLICE: {
    const FGraphNode *a = y->predecessors[0];
    const FSlice *slice = (FSlice *)y->operation.additional_data;
    std::vector<size_t> start(a->operation.dimensions);
    for (int i = 0; i < a->operation.dimensions; i++) {
      start[i] = slice->step[i] >= 0 ? slice->start[i] : slice->end[i] + 1;
    }
    return fextend_step(prev_adj, a->operation.shape, start.data(),
                        slice->step);
  }
  case FEXTEND: {
    const FGraphNode *a = y->predecessors[0];
    const FExtend *extend = (FExtend *)y->operation.additional_data;
    std::vector<long> start(a->operation.dimensions);
    std::vector<long> ends(a->operation.dimensions);
    std::vector<long> steps(a->operation.dimensions);
    for (int i = 0; i < a->operation.dimensions; i++) {
      start[i] = extend->start[i];
      ends[i] = a->operation.shape[i] * extend->step[i] + extend->start[i];
      steps[i] = extend->step[i];
    }
    return fslice_step(prev_adj, start.data(), ends.data(), steps.data());
  }
  case FSIGN:
  case FEVEN:
  case FLESS:
  case FEQUAL:
  case FGREATER:
    return constant_tensor(0.0, F_FLOAT64, y->operation.shape,
                           y->operation.dimensions);
  default:
    flogging(F_WARNING, "No gradient function exists!");
    return nullptr;
  }
}
static void collect(FGraphNode *x, std::list<FGraphNode *> &stack,
                    std::unordered_set<FGraphNode *> &visited,
                    const std::unordered_set<const FGraphNode *> dxs) {
  // TODO could be made more performant with explicit todo stack and a push_back
  // before continuing on the parents
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
void fCalculateGradients(FGraphNode *y, FGraphNode **dx,
                         const unsigned int num_gradients,
                         FGraphNode **gradients) {
  using namespace std;
  unordered_set<const FGraphNode *> *gd =
      (unordered_set<const FGraphNode *> *)y->gradient_data;
  if (!gd)
    flogging(F_ERROR,
             "no derivatives in the operational graph! Don't forget the "
             "necessary calls to fMarkGradientVariable (or in C++ .watch())");
  std::unordered_set<const FGraphNode *> vars(num_gradients);
  for (int i = 0; i < num_gradients; i++) {
    vars.insert(dx[i]);
    if (!gd->contains(dx[i]))
      flogging(F_WARNING,
               "derivative was not marked during graph construction! Don't "
               "forget the "
               "necessary calls to fMarkGradientVariable (or in C++ .watch())");
  }
  // to store gradients per node
  unordered_map<const FGraphNode *, FGraphNode *> adjoints;
  list<FGraphNode *> todo;
  std::unordered_set<FGraphNode *> visited;
  collect(y, todo, visited, vars);
  // initialize
  adjoints[y] = constant_tensor(1., F_FLOAT64, y->operation.shape,
                                y->operation.dimensions);
  while (!todo.empty()) {
    FGraphNode *curr = todo.front();
    todo.pop_front();
    FGraphNode *adj = adjoints[curr];
    for (int i = 0; i < curr->num_predecessor; i++) {
      FGraphNode *parent = curr->predecessors[i];
      if (!visited.contains(parent))
        continue;
    auto start = std::chrono::high_resolution_clock::now();
      FGraphNode *local_grad =
          unbroadcast(local_gradient(curr, i, adj), parent);
      if (adjoints.contains(parent)) {
        adjoints[parent] = fExecuteGraph(fadd(adjoints[parent], local_grad));
      } else {
        adjoints.insert({parent, fExecuteGraph(local_grad)});
      }
     OCLCompilerThread::memory_barrier();
     std::chrono::duration<double, std::milli> elapsed =
         std::chrono::high_resolution_clock::now() - start;
     std::cout << fop_to_string[curr->operation.op_type] << " took " << elapsed.count() << std::endl;
    }
  }
  for (int i = 0; i < num_gradients; i++) {
    if (adjoints.contains(dx[i])) {
      gradients[i] = adjoints[dx[i]];
    } else {
      flogging(F_WARNING, "Operation graph did not contain the derivative!");
      gradients[i] = nullptr;
    }
  }
}
#endif
