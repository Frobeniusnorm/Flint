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

  This file includes methods to pass parameters to the kernels */
#include "../../flint.h"
#include "src/errors.hpp"
#include <CL/cl.h>
#include <list>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <tuple>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>
static cl_mem calcAndPushAccSize(int dim, size_t *shape, cl_kernel kernel,
                                 cl_context context, int &par_index) {
  std::vector<size_t> acc_sizes(dim);
  acc_sizes[dim - 1] = 1;
  for (long d = dim - 2; d >= 0; d--) {
    acc_sizes[d] = acc_sizes[d + 1] * shape[d + 1];
  }
  cl_int err_code;
  cl_mem acc_mem =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     dim * sizeof(long), acc_sizes.data(), &err_code);
  if (!acc_mem) {
    setErrorType(OCL_ERROR);
    flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                          std::to_string(err_code));
    return nullptr;
  }
  if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&acc_mem) !=
      CL_SUCCESS) {
    setErrorType(OCL_ERROR);
    flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                          std::to_string(err_code));
    return nullptr;
  }
  return acc_mem;
}
// values for a single operation (not related directly to parameters)
inline void pushAdditonalVals(FGraphNode *node, cl_kernel kernel,
                              cl_context context, int &par_index,
                              std::list<cl_mem> &to_free) {
  cl_int err_code;
  switch (node->operation.op_type) {
  case FMATMUL: {
    const FGraphNode *gnp1 = node->predecessors[0],
                     *gnp2 = node->predecessors[1];
    long l = gnp1->operation.shape[gnp1->operation.dimensions - 2];
    long m = gnp1->operation.shape[gnp1->operation.dimensions - 1];
    long n = gnp2->operation.shape[gnp2->operation.dimensions - 1];
    for (long *mmd : {&l, &m, &n}) {
      if (clSetKernelArg(kernel, par_index++, sizeof(long), (void *)mmd) !=
          CL_SUCCESS) {
        setErrorType(OCL_ERROR);
        flogging(F_ERROR, "Could not load Argument to kernel!");
        return;
      }
    }
  } break;
  case FREDUCE_MIN:
  case FREDUCE_MAX:
  case FREDUCE_MUL:
  case FREDUCE_SUM: {
    int *dim = ((int *)node->operation.additional_data);
    if (clSetKernelArg(kernel, par_index++, sizeof(int), (void *)dim) !=
        CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
  } break;
  case FSET_INDEX:
  case FINDEX: {
    const FOperation op = node->operation;
    const unsigned int axis =
        node->predecessors[op.op_type == FSET_INDEX ? 2 : 1]
            ->operation.dimensions -
        1;
    size_t acc_sizes_ax = 1;
    for (int i = axis + 1; i < op.dimensions; i++)
      acc_sizes_ax *= op.shape[i];
    // push acc_sizes_ax
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&acc_sizes_ax) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    // push op shape
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&op.shape[axis]) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    if (op.op_type == FSET_INDEX) {
      // push c shape
      if (clSetKernelArg(
              kernel, par_index++, sizeof(long),
              (void *)&node->predecessors[2]->operation.shape[axis]) !=
          CL_SUCCESS) {
        setErrorType(OCL_ERROR);
        flogging(F_ERROR, "Could not load Argument to kernel!");
        return;
      }
    } else {
      // push a shape
      if (clSetKernelArg(
              kernel, par_index++, sizeof(long),
              (void *)&node->predecessors[0]->operation.shape[axis]) !=
          CL_SUCCESS) {
        setErrorType(OCL_ERROR);
        flogging(F_ERROR, "Could not load Argument to kernel!");
        return;
      }
    }
  } break;
  case FGEN_CONSTANT: {
    switch (node->operation.data_type) {
    case F_INT32: {
      int val = ((int *)node->operation.additional_data)[0];
      if (clSetKernelArg(kernel, par_index++, sizeof(int), (void *)&val) !=
          CL_SUCCESS) {
        setErrorType(OCL_ERROR);
        flogging(F_ERROR, "Could not load Argument to kernel!");
        return;
      }
    } break;
    case F_INT64: {
      long val = ((long *)node->operation.additional_data)[0];
      if (clSetKernelArg(kernel, par_index++, sizeof(long), (void *)&val) !=
          CL_SUCCESS) {
        setErrorType(OCL_ERROR);
        flogging(F_ERROR, "Could not load Argument to kernel!");
        return;
      }
    } break;
    case F_FLOAT32: {
      float val = ((float *)node->operation.additional_data)[0];
      if (clSetKernelArg(kernel, par_index++, sizeof(float), (void *)&val) !=
          CL_SUCCESS) {
        setErrorType(OCL_ERROR);
        flogging(F_ERROR, "Could not load Argument to kernel!");
        return;
      }
    } break;
    case F_FLOAT64: {
      double val = ((double *)node->operation.additional_data)[0];
      if (clSetKernelArg(kernel, par_index++, sizeof(double), (void *)&val) !=
          CL_SUCCESS) {
        setErrorType(OCL_ERROR);
        flogging(F_ERROR, "Could not load Argument to kernel!");
        return;
      }
    } break;
    }
  } break;
  case FGEN_RANDOM: {
    // push time parameter
    double seed = ((double *)node->operation.additional_data)[0];
    if (clSetKernelArg(kernel, par_index++, sizeof(double), (void *)&seed) !=
        CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
  } break;
  case FGEN_ARANGE: {
    unsigned int ax = ((unsigned int *)node->operation.additional_data)[0];
    size_t acc_sizes_ax = 1;
    for (unsigned int i = ax + 1; i < node->operation.dimensions; i++)
      acc_sizes_ax *= node->operation.shape[i];
    // push acc_sizes_ax
    if (clSetKernelArg(kernel, par_index++, sizeof(size_t),
                       (void *)&acc_sizes_ax) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    // push shape_ax
    if (clSetKernelArg(kernel, par_index++, sizeof(size_t),
                       (void *)&node->operation.shape[ax]) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
  } break;
  case FCONCAT: {
    // acc_size_last, shape_ax, a_shape_ax, b_shape_ax, ax
    FGraphNode *a = node->predecessors[0];
    FGraphNode *b = node->predecessors[1];
    unsigned int ax = ((unsigned int *)node->operation.additional_data)[0];
    size_t acc_size_last = 1;
    for (int i = node->operation.dimensions - 2; i >= (int)ax; i--)
      acc_size_last *= node->operation.shape[i + 1];
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&acc_size_last) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&(node->operation.shape[ax])) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&a->operation.shape[ax]) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&b->operation.shape[ax]) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    if (clSetKernelArg(kernel, par_index++, sizeof(int), (void *)&ax) !=
        CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
  } break;
  case FGRADIENT_CONVOLVE:
  case FSLIDE: {
    bool is_slide = node->operation.op_type == FSLIDE;
    const FOperation op = node->operation;
    const FGraphNode *gnp1 = node->predecessors[0],
                     *gnp2 = node->predecessors[1];
    FOperation pred;
    FOperation kernel_par;
    FOperation adjoint;
    if (is_slide) {
      pred = gnp1->operation;
      kernel_par = gnp2->operation;
    } else {
      pred = op;
      kernel_par = gnp1->operation;
      adjoint = gnp2->operation;
      // dimensions0
      if (clSetKernelArg(kernel, par_index++, sizeof(int),
                         (void *)&node->operation.dimensions) != CL_SUCCESS) {
        setErrorType(OCL_ERROR);
        flogging(F_ERROR, "Could not load Argument to kernel!");
        return;
      }
    }
    unsigned int *steps = (unsigned int *)op.additional_data;
    // allocate steps
    cl_mem steps_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       op.dimensions * sizeof(int), steps, &err_code);
    if (!steps_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    to_free.push_back(calcAndPushAccSize(pred.dimensions, pred.shape, kernel,
                                         context, par_index));
    to_free.push_back(calcAndPushAccSize(
        kernel_par.dimensions, kernel_par.shape, kernel, context, par_index));
    if (!is_slide) {
      std::vector<size_t> acc_sizes(pred.dimensions - 1);
      acc_sizes[op.dimensions - 2] = 1;
      for (long d = op.dimensions - 3; d >= 0; d--)
        acc_sizes[d] = acc_sizes[d + 1] * adjoint.shape[d + 1];

      cl_mem acc_shape = clCreateBuffer(
          context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          acc_sizes.size() * sizeof(long), acc_sizes.data(), &err_code);
      if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                         (void *)&acc_shape) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                              std::to_string(err_code));
      to_free.push_back(acc_shape);
    }
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&steps_mem) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    const FOperation shape_par = is_slide ? pred : kernel_par;
    const FOperation shape2_par = is_slide ? kernel_par : pred;
    cl_mem shape_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        shape_par.dimensions * sizeof(long), shape_par.shape, &err_code);
    if (!shape_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&shape_mem) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    if (is_slide) {
      cl_mem shape2_mem = clCreateBuffer(
          context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          shape2_par.dimensions * sizeof(long), shape2_par.shape, &err_code);
      if (!shape2_mem)
        flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                              std::to_string(err_code));
      if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                         (void *)&shape2_mem) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                              std::to_string(err_code));
      to_free.push_back(shape2_mem);
    }
    to_free.push_back(steps_mem);
    to_free.push_back(shape_mem);
  } break;
  case FCONVOLVE: {
    const FOperation op = node->operation;
    const FGraphNode *gnp1 = node->predecessors[0],
                     *gnp2 = node->predecessors[1];
    const FOperation pred = gnp1->operation, kernel_par = gnp2->operation;
    unsigned int *steps = (unsigned int *)op.additional_data;
    // allocate steps
    cl_mem steps_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       op.dimensions * sizeof(int), steps, &err_code);
    if (!steps_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    to_free.push_back(calcAndPushAccSize(op.dimensions, op.shape, kernel,
                                         context, par_index));
    to_free.push_back(calcAndPushAccSize(pred.dimensions, pred.shape, kernel,
                                         context, par_index));
    to_free.push_back(calcAndPushAccSize(
        kernel_par.dimensions, kernel_par.shape, kernel, context, par_index));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&steps_mem) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    to_free.push_back(steps_mem);
  } break;
  default:
    break;
  }
}

// parameters per operand
inline void pushParameterVals(FGraphNode *node, FGraphNode *pred,
                              cl_kernel kernel, cl_context context,
                              int &par_index, std::list<cl_mem> &to_free) {
  cl_int err_code;
  FOperation op = pred->operation;
  switch (node->operation.op_type) {
  case FSET_INDEX:
  case FINDEX:
  case FMATMUL:
  case FGRADIENT_CONVOLVE:
  case FSLIDE:
  case FCONVOLVE: {
    if (clSetKernelArg(kernel, par_index++, sizeof(int),
                       (void *)&op.dimensions) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
  } break;
  case FREDUCE_MIN:
  case FREDUCE_MAX:
  case FREDUCE_SUM:
  case FREDUCE_MUL: {
    int dim = ((int *)node->operation.additional_data)[0];
    const FOperation pred = node->predecessors[0]->operation;
    long it_dim = 1; // iteration size <=> product of all dimensions along dim
    for (size_t d = dim + 1; d < pred.dimensions; d++)
      it_dim *= pred.shape[d];
    const long shape_dim = pred.shape[dim];
    if (clSetKernelArg(kernel, par_index++, sizeof(int),
                       (void *)&op.dimensions) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    if (clSetKernelArg(kernel, par_index++, sizeof(long), (void *)&it_dim) !=
        CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    if (clSetKernelArg(kernel, par_index++, sizeof(long), (void *)&shape_dim) !=
        CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
  } break;
  case FTRANSPOSE: {
    if (clSetKernelArg(kernel, par_index++, sizeof(int),
                       (void *)&op.dimensions) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    std::vector<long> acc_sizes_s(op.dimensions);
    acc_sizes_s[op.dimensions - 1] = 1;
    for (int dim = op.dimensions - 2; dim >= 0; dim--) {
      acc_sizes_s[dim] = acc_sizes_s[dim + 1] * op.shape[dim + 1];
    }
    const int *transpositions = (int *)node->operation.additional_data;
    std::vector<long> acc_sizes_st(op.dimensions);
    for (int i = 0; i < op.dimensions; i++) {
      acc_sizes_st[i] = acc_sizes_s[transpositions[i]];
    }
    to_free.push_back(calcAndPushAccSize(node->operation.dimensions,
                                         node->operation.shape, kernel, context,
                                         par_index));
    cl_mem ass_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        op.dimensions * sizeof(long), acc_sizes_st.data(), &err_code);
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&ass_mem) !=
        CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    if (!ass_mem) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    to_free.push_back(ass_mem);
  } break;
  case FSLICE: {
    if (clSetKernelArg(kernel, par_index++, sizeof(int),
                       (void *)&op.dimensions) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    FSlice *slice = (FSlice *)node->operation.additional_data;
    // flattened shape data
    std::vector<size_t> acc_sizes_pred(node->operation.dimensions);
    for (long d = node->operation.dimensions - 1; d >= 0; d--) {
      if (d == node->operation.dimensions - 1)
        acc_sizes_pred[d] = 1;
      else
        acc_sizes_pred[d] = acc_sizes_pred[d + 1] * op.shape[d + 1];
    }
    // allocate steps
    cl_mem steps =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       op.dimensions * sizeof(long), slice->step, &err_code);
    if (!steps)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    // calculate start and step size in flattened array
    long start = 0;
    for (unsigned int d = 0; d < node->operation.dimensions; d++) {
      start += slice->start[d] * acc_sizes_pred[d];
    }
    to_free.push_back(calcAndPushAccSize(node->operation.dimensions,
                                         node->operation.shape, kernel, context,
                                         par_index));
    to_free.push_back(calcAndPushAccSize(op.dimensions, op.shape, kernel,
                                         context, par_index));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&steps) !=
            CL_SUCCESS ||
        clSetKernelArg(kernel, par_index++, sizeof(long), (void *)&start) !=
            CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    to_free.push_back(steps);
  } break;
  case FSLIDING_WINDOW: {
    const FOperation pred = node->predecessors[0]->operation;
    const FSlidingWindow *slidewin =
        (FSlidingWindow *)node->operation.additional_data;
    size_t acc_size = node->operation.shape[1];
    std::vector<size_t> acc_sizes_win(pred.dimensions);
    std::vector<size_t> acc_sizes_rest(pred.dimensions);
    acc_sizes_win[acc_sizes_win.size() - 1] = 1;
    acc_sizes_rest[acc_sizes_win.size() - 1] = 1;
    for (int i = acc_sizes_win.size() - 2; i >= 0; i--) {
      acc_size *= node->operation.shape[i + 2];
      acc_sizes_rest[i] = acc_sizes_rest[i + 1] * slidewin->size[i + 1];
      // no of windows in that dimension
      size_t window_size = pred.shape[i + 1] - slidewin->size[i + 1] + 1;
      window_size = window_size % slidewin->step[i + 1] == 0
                        ? window_size / slidewin->step[i + 1]
                        : window_size / slidewin->step[i + 1] + 1;
      acc_sizes_win[i] = acc_sizes_win[i + 1] * window_size;
    }
    if (clSetKernelArg(kernel, par_index++, sizeof(int),
                       (void *)&op.dimensions) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    cl_mem acc_win_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        pred.dimensions * sizeof(long), acc_sizes_win.data(), &err_code);
    if (!acc_win_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    cl_mem acc_rest_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        pred.dimensions * sizeof(long), acc_sizes_rest.data(), &err_code);
    if (!acc_rest_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    cl_mem steps = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        pred.dimensions * sizeof(unsigned int), slidewin->step, &err_code);
    if (!steps)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    to_free.push_back(calcAndPushAccSize(pred.dimensions, pred.shape, kernel,
                                         context, par_index));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), &acc_win_mem) !=
        CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), &acc_rest_mem) !=
        CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    if (clSetKernelArg(kernel, par_index++, sizeof(long), &acc_size) !=
        CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), &steps) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Arguments to kernel!");
    to_free.push_back(acc_win_mem);
    to_free.push_back(acc_rest_mem);
    to_free.push_back(steps);
  } break;
  case FUNSLIDE_WINDOW: {
    const FOperation pred = node->predecessors[0]->operation;
    unsigned int *steps = (unsigned int *)node->operation.additional_data;
    // dimensions 0
    if (clSetKernelArg(kernel, par_index++, sizeof(int),
                       (void *)&op.dimensions) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    // shapeR
    cl_mem shapeR =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       node->operation.dimensions * sizeof(long),
                       node->operation.shape, &err_code);
    if (!shapeR)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), &shapeR) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel! 0");
    to_free.push_back(shapeR);
    // acc_sizes
    to_free.push_back(calcAndPushAccSize(node->operation.dimensions,
                                         node->operation.shape, kernel, context,
                                         par_index));
    // shapeR
    cl_mem shape0 =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       pred.dimensions * sizeof(long), pred.shape, &err_code);
    if (!shape0)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), &shape0) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel! 1");
    to_free.push_back(shape0);
    // acc_sizes_pred
    to_free.push_back(calcAndPushAccSize(pred.dimensions, pred.shape, kernel,
                                         context, par_index));
    size_t no_windows[pred.dimensions - 1];
    for (int i = 0; i < pred.dimensions - 1; i++) {
      size_t window_size = node->operation.shape[i] - pred.shape[i + 1] + 1;
      no_windows[i] = window_size % steps[i] == 0 ? window_size / steps[i]
                                                  : window_size / steps[i] + 1;
    }
    // acc_no_windows
    to_free.push_back(calcAndPushAccSize(pred.dimensions - 1, no_windows,
                                         kernel, context, par_index));
    // no_windows
    cl_mem windows = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (pred.dimensions - 1) * sizeof(long), no_windows, &err_code);
    if (!windows)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), &windows) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel! 2");
    to_free.push_back(windows);
    // steps
    cl_mem steps_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (pred.dimensions - 1) * sizeof(unsigned int), steps, &err_code);
    if (!steps_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), &steps_mem) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Arguments to kernel! 3");
    to_free.push_back(steps_mem);
  } break;
  case FREPEAT: {
    if (clSetKernelArg(kernel, par_index++, sizeof(int),
                       (void *)&op.dimensions) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    to_free.push_back(calcAndPushAccSize(node->operation.dimensions,
                                         node->operation.shape, kernel, context,
                                         par_index));
    to_free.push_back(calcAndPushAccSize(op.dimensions, op.shape, kernel,
                                         context, par_index));
    cl_mem predshape_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       op.dimensions * sizeof(long), op.shape, &err_code);
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&predshape_mem) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    to_free.push_back(predshape_mem);
  } break;
  case FEXTEND: {
    if (clSetKernelArg(kernel, par_index++, sizeof(int),
                       (void *)&op.dimensions) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    to_free.push_back(calcAndPushAccSize(node->operation.dimensions,
                                         node->operation.shape, kernel, context,
                                         par_index));
    to_free.push_back(calcAndPushAccSize(op.dimensions, op.shape, kernel,
                                         context, par_index));
    const FExtend *extend = (FExtend *)node->operation.additional_data;
    cl_mem steps_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       op.dimensions * sizeof(long), extend->step, &err_code);
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&steps_mem) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    cl_mem start_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       op.dimensions * sizeof(long), extend->start, &err_code);
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&start_mem) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    cl_mem predshape_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       op.dimensions * sizeof(long), op.shape, &err_code);
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&predshape_mem) != CL_SUCCESS) {
      setErrorType(OCL_ERROR);
      flogging(F_ERROR, "Could not load Argument to kernel!");
      return;
    }
    to_free.push_back(steps_mem);
    to_free.push_back(start_mem);
    to_free.push_back(predshape_mem);
  } break;
  default:
    break;
  }
}
/** Returns a map mapping to each node in the subgraph of root the number of
 * incoming edges, i.e. the number of nodes that have that node as parent */
inline std::unordered_map<FGraphNode *, int>
calculateNumEdges(FGraphNode *root) {
  std::list<FGraphNode *> todo;
  std::unordered_map<FGraphNode *, int> num_edges;
  todo.push_front(root);
  num_edges.insert({root, 0});
  while (!todo.empty()) {
    FGraphNode *c = todo.front();
    todo.pop_front();
    for (int i = 0; i < c->num_predecessor; i++) {
      if (num_edges.find(c->predecessors[i]) == num_edges.end()) {
        num_edges.insert({c->predecessors[i], 1});
        todo.push_front(c->predecessors[i]);
      } else
        num_edges[c->predecessors[i]]++;
    }
  }
  return num_edges;
}
/** Calculates a topological sort of the operational graph
 * with Kahns algorithm */
inline std::list<FGraphNode *> topologicalSort(FGraphNode *root) {
  std::list<FGraphNode *> result;
  std::unordered_map<FGraphNode *, int> num_edges = calculateNumEdges(root);
  std::list<FGraphNode *> no_incoming;
  no_incoming.push_back(root);
  while (!no_incoming.empty()) {
    FGraphNode *n = no_incoming.front();
    result.push_back(n);
    no_incoming.pop_front();
    for (int i = 0; i < n->num_predecessor; i++) {
      if (--num_edges[n->predecessors[i]] == 0)
        no_incoming.push_back(n->predecessors[i]);
    }
  }
  return result;
}
