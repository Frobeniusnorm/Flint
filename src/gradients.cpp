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
#include <cmath>
#include <cstring>
#include <iostream>
#include <list>
#include <math.h>
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
// converts c++ type to flint type
template <typename T> static constexpr FType toFlintType() {
  if (std::is_same<T, int>())
    return F_INT32;
  if (std::is_same<T, long>())
    return F_INT64;
  if (std::is_same<T, float>())
    return F_FLOAT32;
  if (std::is_same<T, double>())
    return F_FLOAT64;
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
static FGraphNode *local_gradient(FGraphNode *y, FGraphNode *dx,
                                  FGraphNode *prev_adj) {
  switch (y->operation->op_type) {
  case FSTORE:
  case FCONST:
    return fmul(prev_adj, constant_tensor(
                              y == dx ? 1.0 : 0.0, dx->operation->data_type,
                              dx->operation->shape, dx->operation->dimensions));
  case FADD:
    return fmul(
        prev_adj,
        constant_tensor(
            (dx == y->predecessors[0] || dx == y->predecessors[1]) ? 1.0 : 0.0,
            dx->operation->data_type, dx->operation->shape,
            dx->operation->dimensions));
  case FSUB:
    return fgradient_sub(y->predecessors[0], y->predecessors[1], dx);
  case FMUL: {
    if (y->predecessors[0] == dx) {
      return fmul(prev_adj, y->predecessors[1]);
    } else if (y->predecessors[1] == dx) {
      return fmul(prev_adj, y->predecessors[0]);
    } else
      return nullptr;
  }
  case FDIV:
    return fgradient_div(y->predecessors[0], y->predecessors[1], dx);
  case FPOW:
    return fgradient_pow(y->predecessors[0], y->predecessors[1], dx);
  case FLOG:
    return fgradient_log(y->predecessors[0], dx);
  case FLOG2:
    return fgradient_log2(y->predecessors[0], dx);
  case FLOG10:
    return fgradient_log10(y->predecessors[0], dx);
  case FMATMUL: {
    FGraphNode *a = y->predecessors[0];
    FGraphNode *b = y->predecessors[1];
    if (a == dx) {
      std::vector<int> trans(b->operation->dimensions);
      for (int i = 0; i < trans.size(); i++)
        trans[i] = trans.size() - 1 - i;
      FGraphNode *tb = ftranspose(b, trans.data());
      return fmatmul(prev_adj, tb);
    } else if (b == dx) {
      std::vector<int> trans(a->operation->dimensions);
      for (int i = 0; i < trans.size(); i++)
        trans[i] = trans.size() - 1 - i;
      FGraphNode *ta = ftranspose(a, trans.data());
      return fmatmul(ta, prev_adj);
    } else {
      return nullptr;
    }
  }
  case FLATTEN:
  case FCONVERSION:
  case FRESHAPE:
  case FMIN:
  case FMAX:
  case FREDUCE_SUM:
  case FREDUCE_MUL:
  case FSLICE:
  case FABS:
  case FREPEAT:
  case FTRANSPOSE:
  case FNUM_OPERATION_TYPES:
    return nullptr;
  }
}

static bool adapt_grad = true;
static FGraphNode *unbroadcast(FGraphNode *adjoint, const FGraphNode *node) {
  if (adjoint->operation->dimensions > node->operation->dimensions) {
    size_t diff = adjoint->operation->dimensions - node->operation->dimensions;
    FGraphNode *res = adjoint;
    for (int i = 0; i < diff; i++) {
      res = freduce_sum(res, 0);
    }
    return res;
  }
  return adjoint;
}

FGraphNode *fCalculateGradient(FGraphNode *y, FGraphNode *dx) {
  adapt_grad = false;
  using namespace std;
  // to store gradients per node
  unordered_map<FGraphNode *, FGraphNode *> adjoints;
  // fixpoint iteration
  list<FGraphNode *> working;
  unordered_set<FGraphNode *> in_working;
  working.push_back(y);
  in_working.insert(y);

  // initialize
  adjoints[y] = constant_tensor(1., y->operation->data_type,
                                y->operation->shape, y->operation->dimensions);
  FGraphNode *sol = nullptr;
  while (!working.empty()) {
    FGraphNode *curr = working.front();
    working.pop_front();
    in_working.erase(curr);
    if (curr == dx) {
      sol = adjoints[curr];
      continue;
    }
    if (curr->operation->op_type == FSTORE ||
        curr->operation->op_type == FCONST)
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
  return sol;
}
FGraphNode *fgradient_add(const FGraphNode *x, const FGraphNode *y,
                          const FGraphNode *dx) {
  // ao is the higher dimensional vector
  const FOperation *ao = x->operation;
  const FOperation *bo = y->operation;
  const FGraphNode *a = x, *b = y;
  if (ao->dimensions < bo->dimensions) {
    const FOperation *tmp_op = ao;
    ao = bo;
    bo = tmp_op;
    const FGraphNode *tmp_gn = a;
    a = b;
    b = tmp_gn;
  }
  if (a == dx && b == dx)
    // d(x + x = 2 * x) / dx = 2
    return constant_tensor(2.0, ao->data_type, ao->shape, ao->dimensions);
  else if (b == dx) {
    // count number of times b is added to a and return tensor with shape
    // of b with that value
    int times = 1;
    for (int i = 0; i < ao->dimensions - bo->dimensions; i++)
      times *= ao->shape[i];
    return constant_tensor((double)times, bo->data_type, bo->shape,
                           bo->dimensions);
  } else if (a == dx)
    // return one tensor with shape of a
    return constant_tensor(1.0, ao->data_type, ao->shape, ao->dimensions);
  else
    return constant_tensor(0.0, dx->operation->data_type, dx->operation->shape,
                           dx->operation->dimensions);
}
FGraphNode *fgradient_sub(const FGraphNode *x, const FGraphNode *y,
                          const FGraphNode *dx) {
  // ao is the higher dimensional vector
  const FOperation *ao = x->operation;
  const FOperation *bo = y->operation;
  const FGraphNode *a = x, *b = y;
  bool switched = false;
  if (ao->dimensions < bo->dimensions) {
    const FOperation *tmp_op = ao;
    ao = bo;
    bo = tmp_op;
    const FGraphNode *tmp_gn = a;
    a = b;
    b = tmp_gn;
    switched = true;
  }
  if (a == dx && b == dx)
    // d(x - x = x) / dx = 0
    return constant_tensor(0.0, ao->data_type, ao->shape, ao->dimensions);
  else if (b == dx) {
    // count number of times b is added to a and return tensor with shape
    // of b with that value
    int times = 1;
    for (int i = 0; i < ao->dimensions - bo->dimensions; i++)
      times *= ao->shape[i];
    return constant_tensor((switched ? 1 : -1) * (double)times, bo->data_type,
                           bo->shape, bo->dimensions);
  } else if (a == dx)
    // return one tensor with shape of a
    return constant_tensor((double)(switched ? -1 : 1), ao->data_type,
                           ao->shape, ao->dimensions);
  else
    return constant_tensor(0.0, dx->operation->data_type, dx->operation->shape,
                           dx->operation->dimensions);
}
FGraphNode *fgradient_mul(FGraphNode *x, FGraphNode *y, const FGraphNode *dx) {
  // ao is the higher dimensional vector
  FOperation *ao = x->operation;
  FOperation *bo = y->operation;
  FGraphNode *a = x, *b = y;
  if (ao->dimensions < bo->dimensions) {
    FOperation *tmp_op = ao;
    ao = bo;
    bo = tmp_op;
    FGraphNode *tmp_gn = a;
    a = b;
    b = tmp_gn;
  }
  if (b == dx) {
    FGraphNode *result = a;
    if (!adapt_grad)
      return result;
    // dx is the smaller one -> reduce_sum other one
    for (int dim = 0; dim < ao->dimensions - bo->dimensions; dim++)
      result = freduce_sum(result, dim);
    return result;
  } else if (a == dx) {
    // const is special case since it technically has an incompatible shape
    if (bo->op_type == FCONST) {
      FConst *co = (FConst *)bo->additional_data;
      switch (bo->data_type) {
      case F_INT32:
        return fconstant_i(*((int *)co->value), ao->shape, ao->dimensions);
      case F_INT64:
        return fconstant_l(*((long *)co->value), ao->shape, ao->dimensions);
      case F_FLOAT32:
        return fconstant_f(*((float *)co->value), ao->shape, ao->dimensions);
      case F_FLOAT64:
        return fconstant_d(*((double *)co->value), ao->shape, ao->dimensions);
      }
    }
    // if not const
    FGraphNode *result = b;
    if (!adapt_grad)
      return result;
    // dx is the bigger one -> expand other one
    size_t new_shape[ao->dimensions];
    int repetitions[ao->dimensions];
    for (int i = 0; i < ao->dimensions - bo->dimensions; i++) {
      new_shape[i] = 1;
      repetitions[i] = ao->shape[i] - 1;
    }
    memcpy(new_shape + (ao->dimensions - bo->dimensions), bo->shape,
           bo->dimensions * sizeof(size_t));
    memset(repetitions + (ao->dimensions - bo->dimensions), 0,
           bo->dimensions * sizeof(int));
    result = freshape(result, new_shape, ao->dimensions);
    result = frepeat(result, repetitions);
    return result;
  } else
    return constant_tensor(0.0, dx->operation->data_type, dx->operation->shape,
                           dx->operation->dimensions);
}
FGraphNode *fgradient_div(FGraphNode *a, FGraphNode *b, const FGraphNode *dx) {
  const FOperation *ao = a->operation;
  const FOperation *bo = b->operation;
  if (a == dx) {
    FGraphNode *opb;
    if (bo->op_type == FCONST) {
      FConst *bop = (FConst *)bo->additional_data;
      double div_val;
      switch (bo->data_type) {
      case F_INT32:
        div_val = 1.0 / *((int *)bop->value);
        break;
      case F_INT64:
        div_val = 1.0 / *((long *)bop->value);
        break;
      case F_FLOAT32:
        div_val = 1.0 / *((float *)bop->value);
        break;
      case F_FLOAT64:
        div_val = 1.0 / *((double *)bop->value);
        break;
      }
      opb = constant_tensor(div_val, F_FLOAT64, ao->shape, ao->dimensions);
    } else
      opb = fdiv_g(
          constant_tensor(1., bo->data_type, bo->shape, bo->dimensions), b);
    // (a / b) / da = (a * 1 / b) / da
    return fgradient_mul(a, opb, a);
  } else if (b == dx) {
    // (a / b) / db = (a * b^-1) / db = -a * b^-2
    FGraphNode *result = fmul(fmul(a, -1), fpow(b, -2));
    if (result->operation->data_type != dx->operation->data_type)
      result = fconvert(result, dx->operation->data_type);
    if (bo->dimensions < ao->dimensions) {
      for (int dim = 0; dim < ao->dimensions - bo->dimensions; dim++)
        result = freduce_sum(result, dim);
    }
    return result;
  } else
    return constant_tensor(0.0, dx->operation->data_type, dx->operation->shape,
                           dx->operation->dimensions);
}
FGraphNode *fgradient_pow(FGraphNode *a, FGraphNode *b, const FGraphNode *dx) {
  if (a == dx) {
    // x^b / dx = b*x^(b-1)
    return fmul(b, fpow(a, fsub(b, 1)));
  } else if (b == dx) {
    // b^x / dx = b^x * ln(x)
    return fmul(fpow(a, b), flog(b));
  } else
    return constant_tensor(0.0, dx->operation->data_type, dx->operation->shape,
                           dx->operation->dimensions);
}
FGraphNode *fgradient_log(FGraphNode *a, FGraphNode *dx) {
  if (a == dx)
    return fdiv(1.0, a);
  else
    return constant_tensor(0.0, dx->operation->data_type, dx->operation->shape,
                           dx->operation->dimensions);
}
FGraphNode *fgradient_log10(FGraphNode *a, FGraphNode *dx) {
  if (a == dx)
    return fdiv(1.0, fmul(a, log(10.0)));
  else
    return constant_tensor(0.0, dx->operation->data_type, dx->operation->shape,
                           dx->operation->dimensions);
}
FGraphNode *fgradient_log2(FGraphNode *a, FGraphNode *dx) {
  if (a == dx)
    return fdiv(1.0, fmul(a, log(2.0)));
  else
    return constant_tensor(0.0, dx->operation->data_type, dx->operation->shape,
                           dx->operation->dimensions);
}
FGraphNode *fgradient_matmul(FGraphNode *a, FGraphNode *b, FGraphNode *dx) {
  const FOperation *ao = a->operation;
  const FOperation *bo = b->operation;
  if (a == dx) {
    FGraphNode *onetensor =
        constant_tensor(1.0, ao->data_type, ao->shape, ao->dimensions);
    if (ao->dimensions == bo->dimensions) {
      return fmatmul(onetensor, b);
    } else if (bo->dimensions > ao->dimensions) {
      //  dim(b) > dim(a) -> reduce_sum(transpose(matmul(b, 1-tensor)), axis =
      //  -1)
      std::vector<int> transpositions(bo->dimensions);
      // only the last common dimensions have to be transposed
      int start = bo->dimensions - ao->dimensions;
      for (int i = 0; i < start; i++)
        transpositions[i] = i;
      for (int i = 0; start + i < bo->dimensions; i++)
        transpositions[start + i] = bo->dimensions - 1 - i;
      FGraphNode *result =
          ftranspose(fmatmul(b, onetensor), transpositions.data());
      if (!adapt_grad)
        return result;
      for (int dim = 0; dim < bo->dimensions - ao->dimensions; dim++)
        result = freduce_sum(result, dim);
      return result;
    } else {
      // dim(a) > dim(b) -> repeat(transpose(matmul(b, 1-tensor)), axis=0)
      std::vector<int> transpositions(ao->dimensions);
      // only the last common dimensions have to be transposed
      int start = ao->dimensions - bo->dimensions;
      for (int i = 0; i < start; i++)
        transpositions[i] = i;
      for (int i = 0; start + i < ao->dimensions; i++)
        transpositions[start + i] = ao->dimensions - 1 - i;
      FGraphNode *result =
          ftranspose(fmatmul(b, onetensor), transpositions.data());
      if (!adapt_grad)
        return result;
      size_t new_shape[ao->dimensions];
      int repetitions[ao->dimensions];
      for (int i = 0; i < ao->dimensions - bo->dimensions; i++) {
        new_shape[i] = 1;
        repetitions[i] = ao->shape[i] - 1;
      }
      memcpy(new_shape + (ao->dimensions - bo->dimensions), bo->shape,
             bo->dimensions * sizeof(size_t));
      memset(repetitions + (ao->dimensions - bo->dimensions), 0,
             bo->dimensions * sizeof(int));
      result = freshape(result, new_shape, ao->dimensions);
      result = frepeat(result, repetitions);
      return result;
    }
  } else if (b == dx) {
    FGraphNode *onetensor =
        constant_tensor(1.0, bo->data_type, bo->shape, bo->dimensions);
    //  same shape -> a matmul with 1-tensor
    if (ao->dimensions == bo->dimensions) {
      return fmatmul(a, onetensor);
    } else if (bo->dimensions > ao->dimensions) {
      //  dim(b) > dim(a) -> repeat(transpose(matmul(1-tensor, a)), axis=0)
      std::vector<int> transpositions(bo->dimensions);
      // only the last common dimensions have to be transposed
      int start = bo->dimensions - ao->dimensions;
      for (int i = 0; i < start; i++)
        transpositions[i] = i;
      for (int i = 0; start + i < bo->dimensions; i++)
        transpositions[start + i] = bo->dimensions - 1 - i;
      FGraphNode *result =
          ftranspose(fmatmul(onetensor, a), transpositions.data());
      if (!adapt_grad)
        return result;
      size_t new_shape[bo->dimensions];
      int repetitions[bo->dimensions];
      for (int i = 0; i < bo->dimensions - ao->dimensions; i++) {
        new_shape[i] = 1;
        repetitions[i] = bo->shape[i] - 1;
      }
      memcpy(new_shape + (bo->dimensions - ao->dimensions), ao->shape,
             ao->dimensions * sizeof(size_t));
      memset(repetitions + (bo->dimensions - ao->dimensions), 0,
             ao->dimensions * sizeof(int));
      result = freshape(result, new_shape, bo->dimensions);
      result = frepeat(result, repetitions);
      return result;
    } else {
      //  dim(a) > dim(b) -> reduce_sum(transpose(matmul(1-tensor, a)), axis =
      //  -1)
      std::vector<int> transpositions(ao->dimensions);
      // only the last common dimensions have to be transposed
      int start = ao->dimensions - bo->dimensions;
      for (int i = 0; i < start; i++)
        transpositions[i] = i;
      for (int i = 0; start + i < ao->dimensions; i++)
        transpositions[start + i] = ao->dimensions - 1 - i;
      FGraphNode *result =
          ftranspose(fmatmul(onetensor, a), transpositions.data());
      if (!adapt_grad)
        return result;
      for (int dim = 0; dim < ao->dimensions - bo->dimensions; dim++)
        result = freduce_sum(result, dim);
      return result;
    }
  } else
    return constant_tensor(0.0, dx->operation->data_type, dx->operation->shape,
                           dx->operation->dimensions);
}
#endif
