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
          CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
    }
  } break;
  case FREDUCE_MIN:
  case FREDUCE_MAX:
  case FREDUCE_MUL:
  case FREDUCE_SUM: {
    int *dim = ((int *)node->operation.additional_data);
    if (clSetKernelArg(kernel, par_index++, sizeof(int), (void *)dim) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
  } break;
  case FMULTI_INDEX:
  case FINDEX: {
    const unsigned int axis = node->predecessors[1]->operation.dimensions - 1;
    const FOperation op = node->operation;
    size_t acc_sizes_ax = 1;
    for (int i = axis + 1; i < op.dimensions; i++)
      acc_sizes_ax *= op.shape[i];
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&acc_sizes_ax) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&op.shape[axis]) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&node->predecessors[0]->operation.shape[axis]) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
  } break;
  case FGEN_CONSTANT: {
    switch (node->operation.data_type) {
    case F_INT32: {
      int val = ((int *)node->operation.additional_data)[0];
      if (clSetKernelArg(kernel, par_index++, sizeof(int), (void *)&val) !=
          CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
    } break;
    case F_INT64: {
      long val = ((long *)node->operation.additional_data)[0];
      if (clSetKernelArg(kernel, par_index++, sizeof(long), (void *)&val) !=
          CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
    } break;
    case F_FLOAT32: {
      float val = ((float *)node->operation.additional_data)[0];
      if (clSetKernelArg(kernel, par_index++, sizeof(float), (void *)&val) !=
          CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
    } break;
    case F_FLOAT64: {
      double val = ((double *)node->operation.additional_data)[0];
      if (clSetKernelArg(kernel, par_index++, sizeof(double), (void *)&val) !=
          CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
    } break;
    }
  } break;
  case FGEN_RANDOM: {
    // push time parameter
    std::chrono::duration<double, std::nano> tm =
        std::chrono::high_resolution_clock::now().time_since_epoch();
    double t = ((unsigned long)tm.count() % 1000000) / 100.0;
    if (clSetKernelArg(kernel, par_index++, sizeof(double), (void *)&t) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
  } break;
  case FCONCAT: {
    // acc_size_last, shape_ax, a_shape_ax, b_shape_ax, ax
    FGraphNode *a = node->predecessors[0];
    FGraphNode *b = node->predecessors[1];
    unsigned int ax = ((unsigned int *)node->operation.additional_data)[0];
    size_t acc_size_last = 1;
    for (int i = node->operation.dimensions - 2; i >= (int)ax; i--) {
      acc_size_last *= node->operation.shape[i + 1];
    }
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&acc_size_last) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&(node->operation.shape[ax])) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&a->operation.shape[ax]) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&b->operation.shape[ax]) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    if (clSetKernelArg(kernel, par_index++, sizeof(int), (void *)&ax) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
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
                         (void *)&node->operation.dimensions) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
    }
    unsigned int *steps = (unsigned int *)op.additional_data;
    // calculate accumulated sizes for result, kernel and source (pred)
    std::vector<size_t> acc_sizes_pred(pred.dimensions);
    std::vector<size_t> acc_sizes_kernel(kernel_par.dimensions);
    acc_sizes_pred[pred.dimensions - 1] = 1;
    acc_sizes_kernel[kernel_par.dimensions - 1] = 1;
    for (long d = pred.dimensions - 2; d >= 0; d--) {
      acc_sizes_pred[d] = acc_sizes_pred[d + 1] * pred.shape[d + 1];
      acc_sizes_kernel[d] = acc_sizes_kernel[d + 1] * kernel_par.shape[d + 1];
    }
    cl_mem acc_pred_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        pred.dimensions * sizeof(long), acc_sizes_pred.data(), &err_code);
    if (!acc_pred_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    cl_mem acc_kernel_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       kernel_par.dimensions * sizeof(long),
                       acc_sizes_kernel.data(), &err_code);
    if (!acc_kernel_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    // allocate steps
    cl_mem steps_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       op.dimensions * sizeof(int), steps, &err_code);
    if (!steps_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&acc_pred_mem) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&acc_kernel_mem) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
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
    to_free.push_back(acc_pred_mem);
    to_free.push_back(acc_kernel_mem);
    to_free.push_back(steps_mem);
    to_free.push_back(shape_mem);
  } break;
  case FCONVOLVE: {
    const FOperation op = node->operation;
    const FGraphNode *gnp1 = node->predecessors[0],
                     *gnp2 = node->predecessors[1];
    const FOperation pred = gnp1->operation, kernel_par = gnp2->operation;
    unsigned int *steps = (unsigned int *)op.additional_data;
    // calculate accumulated sizes for result, kernel and source (pred)
    std::vector<size_t> acc_sizes(op.dimensions);
    std::vector<size_t> acc_sizes_pred(acc_sizes.size() + 1);
    std::vector<size_t> acc_sizes_kernel(acc_sizes.size() + 1);
    acc_sizes[op.dimensions - 1] = 1;
    for (long d = op.dimensions - 2; d >= 0; d--) {
      acc_sizes[d] = acc_sizes[d + 1] * op.shape[d + 1];
    }
    acc_sizes_kernel[acc_sizes.size()] = 1;
    acc_sizes_pred[acc_sizes.size()] = 1;
    for (long d = acc_sizes.size() - 1; d >= 0; d--) {
      acc_sizes_kernel[d] = acc_sizes_kernel[d + 1] * kernel_par.shape[d + 1];
      acc_sizes_pred[d] = acc_sizes_pred[d + 1] * pred.shape[d + 1];
    }
    cl_mem acc_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        op.dimensions * sizeof(long), acc_sizes.data(), &err_code);
    if (!acc_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    cl_mem acc_pred_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        pred.dimensions * sizeof(long), acc_sizes_pred.data(), &err_code);
    if (!acc_pred_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    cl_mem acc_kernel_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       kernel_par.dimensions * sizeof(long),
                       acc_sizes_kernel.data(), &err_code);
    if (!acc_kernel_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    // allocate steps
    cl_mem steps_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       op.dimensions * sizeof(int), steps, &err_code);
    if (!steps_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&acc_mem) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&acc_pred_mem) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&acc_kernel_mem) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&steps_mem) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    to_free.push_back(acc_mem);
    to_free.push_back(acc_pred_mem);
    to_free.push_back(acc_kernel_mem);
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
  case FMULTI_INDEX:
  case FINDEX:
  case FMATMUL:
  case FGRADIENT_CONVOLVE:
  case FSLIDE:
  case FCONVOLVE: {
    if (clSetKernelArg(kernel, par_index++, sizeof(int),
                       (void *)&op.dimensions) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
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
                       (void *)&op.dimensions) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    if (clSetKernelArg(kernel, par_index++, sizeof(long), (void *)&it_dim) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    if (clSetKernelArg(kernel, par_index++, sizeof(long), (void *)&shape_dim) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
  } break;
  case FTRANSPOSE: {
    if (clSetKernelArg(kernel, par_index++, sizeof(int),
                       (void *)&op.dimensions) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    std::vector<long> acc_sizes_d(op.dimensions);
    std::vector<long> acc_sizes_s(op.dimensions);
    acc_sizes_d[op.dimensions - 1] = 1;
    acc_sizes_s[op.dimensions - 1] = 1;
    for (int dim = op.dimensions - 2; dim >= 0; dim--) {
      acc_sizes_d[dim] = acc_sizes_d[dim + 1] * node->operation.shape[dim + 1];
      acc_sizes_s[dim] = acc_sizes_s[dim + 1] * op.shape[dim + 1];
    }
    const int *transpositions = (int *)node->operation.additional_data;
    std::vector<long> acc_sizes_st(op.dimensions);
    for (int i = 0; i < op.dimensions; i++) {
      acc_sizes_st[i] = acc_sizes_s[transpositions[i]];
    }
    cl_mem asd_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        op.dimensions * sizeof(long), acc_sizes_d.data(), &err_code);
    if (!asd_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&asd_mem) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    cl_mem ass_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        op.dimensions * sizeof(long), acc_sizes_st.data(), &err_code);
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&ass_mem) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    if (!ass_mem)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    to_free.push_back(asd_mem);
    to_free.push_back(ass_mem);
  } break;
  case FSLICE: {
    if (clSetKernelArg(kernel, par_index++, sizeof(int),
                       (void *)&op.dimensions) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    FSlice *slice = (FSlice *)node->operation.additional_data;
    // flattened shape data
    std::vector<size_t> acc_sizes(node->operation.dimensions);
    std::vector<size_t> acc_sizes_pred(acc_sizes.size());
    for (long d = node->operation.dimensions - 1; d >= 0; d--) {
      if (d == node->operation.dimensions - 1) {
        acc_sizes[d] = 1;
        acc_sizes_pred[d] = 1;
      } else {
        acc_sizes_pred[d] = acc_sizes_pred[d + 1] * op.shape[d + 1];
        acc_sizes[d] = acc_sizes[d + 1] * node->operation.shape[d + 1];
      }
    }
    cl_mem acc_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        op.dimensions * sizeof(long), acc_sizes.data(), &err_code);
    if (!acc_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    cl_mem acc_pred_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        op.dimensions * sizeof(long), acc_sizes_pred.data(), &err_code);
    if (!acc_pred_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
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
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&acc_mem) !=
            CL_SUCCESS ||
        clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&acc_pred_mem) != CL_SUCCESS ||
        clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&steps) !=
            CL_SUCCESS ||
        clSetKernelArg(kernel, par_index++, sizeof(long), (void *)&start) !=
            CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    to_free.push_back(acc_mem);
    to_free.push_back(acc_pred_mem);
    to_free.push_back(steps);
  } break;
  case FREPEAT: {
    if (clSetKernelArg(kernel, par_index++, sizeof(int),
                       (void *)&op.dimensions) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    std::vector<long> acc_sizes_d(op.dimensions);
    std::vector<long> acc_sizes_s(op.dimensions);
    acc_sizes_d[op.dimensions - 1] = 1;
    acc_sizes_s[op.dimensions - 1] = 1;
    for (int dim = op.dimensions - 2; dim >= 0; dim--) {
      acc_sizes_d[dim] = acc_sizes_d[dim + 1] * node->operation.shape[dim + 1];
      acc_sizes_s[dim] = acc_sizes_s[dim + 1] * op.shape[dim + 1];
    }
    cl_mem asd_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        op.dimensions * sizeof(long), acc_sizes_d.data(), &err_code);
    if (!asd_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&asd_mem) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    cl_mem ass_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        op.dimensions * sizeof(long), acc_sizes_s.data(), &err_code);
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&ass_mem) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    if (!ass_mem)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    cl_mem predshape_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       op.dimensions * sizeof(long), op.shape, &err_code);
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&predshape_mem) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    to_free.push_back(asd_mem);
    to_free.push_back(ass_mem);
    to_free.push_back(predshape_mem);
  } break;
  case FEXTEND: {
    if (clSetKernelArg(kernel, par_index++, sizeof(int),
                       (void *)&op.dimensions) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    std::vector<long> acc_sizes_d(op.dimensions);
    std::vector<long> acc_sizes_s(op.dimensions);
    acc_sizes_d[op.dimensions - 1] = 1;
    acc_sizes_s[op.dimensions - 1] = 1;
    for (int dim = op.dimensions - 2; dim >= 0; dim--) {
      acc_sizes_d[dim] = acc_sizes_d[dim + 1] * node->operation.shape[dim + 1];
      acc_sizes_s[dim] = acc_sizes_s[dim + 1] * op.shape[dim + 1];
    }
    cl_mem asd_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        op.dimensions * sizeof(long), acc_sizes_d.data(), &err_code);
    if (!asd_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&asd_mem) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    cl_mem ass_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        op.dimensions * sizeof(long), acc_sizes_s.data(), &err_code);
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&ass_mem) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    if (!ass_mem)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    const FExtend *extend = (FExtend *)node->operation.additional_data;
    cl_mem steps_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       op.dimensions * sizeof(long), extend->step, &err_code);
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&steps_mem) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    cl_mem start_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       op.dimensions * sizeof(long), extend->start, &err_code);
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&start_mem) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    cl_mem predshape_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       op.dimensions * sizeof(long), op.shape, &err_code);
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                       (void *)&predshape_mem) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    to_free.push_back(asd_mem);
    to_free.push_back(ass_mem);
    to_free.push_back(steps_mem);
    to_free.push_back(start_mem);
    to_free.push_back(predshape_mem);
  } break;
  default:
    break;
  }
}
