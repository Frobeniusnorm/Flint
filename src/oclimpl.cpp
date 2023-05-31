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

  This file includes the implementation of the GPU backend.
*/

#include "../flint.h"
#include "ocl_codegen.hpp"
#include "ocl_comp.hpp"
#include "utils.hpp"
#include <CL/cl.h>
#include <iostream>
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
static const char *clCompilerOpts = "-cl-no-signed-zeros";
static void openclCallback(const char *errinfo, const void *privateinfo,
                           size_t cb, void *user_data) {
  flogging(F_WARNING, "{OpenCL} " + std::string(errinfo));
}

static bool initialized = false;
// opencl vars
static cl_context context;
static cl_command_queue clqueue;
static cl_device_id device;

void flintInit_gpu() {
  cl_platform_id platforms[10];
  cl_uint num_dev, num_plat;
  if (clGetPlatformIDs(10, &platforms[0], &num_plat) != CL_SUCCESS)
    flogging(F_ERROR, "clGetPlatformIds");
  if (num_plat == 0)
    flogging(F_ERROR,
             "Could not find any OpenCL Platform available! Please make "
             "sure, you have setup your OpenCL driver right!");
  flogging(F_VERBOSE, "Found " + std::to_string(num_plat) + " platforms!");
  device = NULL;
  // find suitable device
  char dev_name[128];
  size_t dev_name_size;
  char dev_vers[128];
  size_t dev_vers_size;
  char dev_vend[128];
  size_t dev_vend_size;
  cl_device_type dev_type;
  cl_device_type highest_type = 0;
  size_t dev_type_size;
  std::string dev_type_string;
  for (int i = 0; i < num_plat; i++) {
    cl_device_id curr_dev;
    if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_DEFAULT, 1, &curr_dev,
                       &num_dev) != CL_SUCCESS) {
      flogging(F_WARNING, "clGetDeviceIDS did not return CL_SUCCESS!");
      continue;
    }
    if (num_dev == 0) {
      flogging(F_WARNING, "Platform has no devices!");
      continue;
    }
    clGetDeviceInfo(curr_dev, CL_DEVICE_NAME, 128 * sizeof(char),
                    (void *)&dev_name[0], &dev_name_size);
    clGetDeviceInfo(curr_dev, CL_DEVICE_VERSION, 128, (void *)&dev_vers[0],
                    &dev_vers_size);
    clGetDeviceInfo(curr_dev, CL_DEVICE_VENDOR, 128, (void *)&dev_vend[0],
                    &dev_vend_size);
    clGetDeviceInfo(curr_dev, CL_DEVICE_TYPE, sizeof(dev_type),
                    (void *)&dev_type, &dev_type_size);
    if (dev_type > highest_type) {
      highest_type = dev_type;
      device = curr_dev;
      if ((dev_type & CL_DEVICE_TYPE_CPU) == CL_DEVICE_TYPE_CPU) {
        dev_type_string = "CPU";
      } else if ((dev_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU) {
        dev_type_string = "GPU";
      } else if ((dev_type & CL_DEVICE_TYPE_ACCELERATOR) ==
                 CL_DEVICE_TYPE_ACCELERATOR) {
        dev_type_string = "Accelerator";
      } else
        dev_type_string = "Device";
    }
  }
  if (!device) {
    flogging(F_ERROR,
             "Could not find any OpenCL devices available! Please make sure, "
             "you have setup your OpenCL driver right!");
  }
  std::string info = "Using " + dev_type_string + " '" +
                     std::string(dev_vend, dev_vend_size - 1) + "', '" +
                     std::string(dev_name, dev_name_size - 1) +
                     "' with OpenCL version " +
                     std::string(dev_vers, dev_vers_size);
  flogging(F_INFO, info);
  int status = 0;
  context = clCreateContext(NULL, 1, &device, openclCallback, NULL, &status);
  if (status != CL_SUCCESS) {
    std::string err = "Could not create OpenCL context: ";
    switch (status) {
    case CL_INVALID_VALUE:
      err += "invalid value";
      break;
    case CL_INVALID_DEVICE:
      err += "invalid device";
      break;
    case CL_DEVICE_NOT_AVAILABLE:
      err += "device is not available";
      break;
    case CL_OUT_OF_RESOURCES:
      err += "out of resources";
      break;
    case CL_OUT_OF_HOST_MEMORY:
      err += "out of host memory";
      break;
    }
    flogging(F_ERROR, err);
  }
  clqueue = clCreateCommandQueueWithProperties(context, device, NULL, &status);
  if (status != CL_SUCCESS)
    flogging(F_ERROR, "clCreateCommandQueue");
  initialized = true;
  flogging(F_VERBOSE, "Flint GPU backend was initialized!");
}
static cl_mem create_gpu_memory(FGraphNode *node, cl_mem_flags memory_type,
                                size_t *total_size = nullptr) {
  cl_int err_code;
  size_t type_size_node = typeSize(node->operation->data_type);
  size_t total_size_node = 1;
  for (int i = 0; i < node->operation->dimensions; i++)
    total_size_node *= node->operation->shape[i];
  const cl_mem result_mem =
      clCreateBuffer(context, memory_type, total_size_node * type_size_node,
                     nullptr, &err_code);
  if (err_code == CL_OUT_OF_HOST_MEMORY)
    flogging(F_ERROR, "Not enough memory to create buffer!");
  if (err_code != CL_SUCCESS)
    flogging(F_ERROR, "Unknown Error while creating gpu memory!");
  if (total_size)
    *total_size = total_size_node;
  return result_mem;
}
#include <chrono>
#include <unordered_map>
cl_kernel OCLCompilerThread::eager_compile(FGraphNode *node, int hash) {
  cl_int err_code;
  cl_kernel kernel = nullptr;
  auto start = std::chrono::high_resolution_clock::now();
  // generate code for this operation for all datatypes
  std::vector<FType> par_types(node->num_predecessor);
  std::string code;
  std::string our_kernel;
  std::vector<std::pair<int, std::string>> all_kernels;
  switch (node->operation->op_type) {
  case FEVEN:
  case FCONVERSION: { // depends on operation
    for (int i = 0; i < node->num_predecessor; i++)
      par_types[i] = node->predecessors[i]->operation->data_type;
    code = generateEagerCode(node->operation->op_type,
                             node->operation->data_type, par_types, our_kernel);
    all_kernels.push_back({hash, our_kernel});
  } break;
  case FGEN_RANDOM: {
    code = generateEagerCode(node->operation->op_type,
                             node->operation->data_type, {}, our_kernel);
    all_kernels.push_back({hash, our_kernel});
  } break;
  case FSIGN:
  case FEQUAL:
  case FLESS:
  case FGREATER: { // result is always FINT32
    std::vector<std::vector<FType>> par_poss =
        allTypePermutations(node->num_predecessor);
    for (std::vector<FType> &params : par_poss) {
      std::string kernel_name;
      code += generateEagerCode(node->operation->op_type, F_INT32, params,
                                kernel_name);
      bool correct_one = true;
      for (int i = 0; i < node->num_predecessor; i++)
        if (params[i] != node->predecessors[i]->operation->data_type) {
          correct_one = false;
          break;
        }
      if (correct_one)
        our_kernel = kernel_name;
      all_kernels.push_back({OCLCompilerThread::generateKernelHash(
                                 node->operation->op_type, F_INT32, params),
                             kernel_name});
    }
  } break;
  case FSQRT:
  case FLOG:
  case FLOG2:
  case FLOG10:
  case FSIN:
  case FCOS:
  case FTAN:
  case FASIN:
  case FACOS:
  case FATAN: {
    for (FType param : {F_FLOAT32, F_FLOAT64}) {
      std::string kernel_name;
      code += generateEagerCode(node->operation->op_type, param, {param},
                                kernel_name);
      bool correct_one = true;
      for (int i = 0; i < node->num_predecessor; i++)
        if (param != node->predecessors[i]->operation->data_type) {
          correct_one = false;
          break;
        }
      if (correct_one)
        our_kernel = kernel_name;
      all_kernels.push_back({OCLCompilerThread::generateKernelHash(
                                 node->operation->op_type, param, {param}),
                             kernel_name});
    }
    break;
  }
  case FGRADIENT_CONVOLVE: {
    std::string kernel_name;
    for (FType param : {F_INT32, F_INT64, F_FLOAT32, F_FLOAT64}) {
      code += generateEagerCode(node->operation->op_type, F_FLOAT64,
                                {param, F_FLOAT64}, kernel_name);
      if (param == node->predecessors[0]->operation->data_type)
        our_kernel = kernel_name;
      all_kernels.push_back(
          {OCLCompilerThread::generateKernelHash(node->operation->op_type,
                                                 F_FLOAT64, {param, F_FLOAT64}),
           kernel_name});
    }
  } break;
  default: {
    std::vector<std::vector<FType>> par_poss =
        allTypePermutations(node->num_predecessor);
    for (std::vector<FType> &params : par_poss) {
      std::string kernel_name;
      FType highest = F_INT32;
      bool correct_one = true;
      for (int i = 0; i < node->num_predecessor; i++) {
        if (params[i] != node->predecessors[i]->operation->data_type)
          correct_one = false;
        highest = higherType(params[i], highest);
      }
      code += generateEagerCode(node->operation->op_type, highest, params,
                                kernel_name);
      if (correct_one)
        our_kernel = kernel_name;
      all_kernels.push_back({OCLCompilerThread::generateKernelHash(
                                 node->operation->op_type, highest, params),
                             kernel_name});
    }
  } break;
  }
  flogging(F_DEBUG, std::string("Eager Kernel Generation for ") +
                        fop_to_string[node->operation->op_type] + ": " + code);
  // generate kernel
  const char *code_data = code.data();
  const size_t code_length = code.length();
  cl_program prog = clCreateProgramWithSource(context, 1, &code_data,
                                              &code_length, &err_code);
  if (err_code == CL_OUT_OF_RESOURCES)
    flogging(F_ERROR, "Out of resources while creating program!");
  if (err_code == CL_OUT_OF_HOST_MEMORY)
    flogging(F_ERROR, "Not enough memory to create program!");
  // build program
  err_code = clBuildProgram(prog, 1, &device, clCompilerOpts, nullptr, nullptr);
  if (err_code == CL_INVALID_PROGRAM)
    flogging(F_ERROR,
             "Invalid Program was generated! Generated code: \"\n" + code +
                 "\"\nPlease contact a developer and/or file a bug report.");
  else if (err_code == CL_COMPILER_NOT_AVAILABLE)
    flogging(F_ERROR, "Compiler of your GPU driver is not available!");
  else if (err_code == CL_OUT_OF_HOST_MEMORY)
    flogging(F_ERROR, "Not enough memory to build program!");
  else if (err_code != CL_SUCCESS) {
    char build_log[4096];
    size_t actual_size = 0;
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 4096,
                          (void *)&build_log[0], &actual_size);
    flogging(F_ERROR,
             "Unknown Error during program compilation! Generated code: \"\n" +
                 code + "\nBuild Log:\n" + std::string(&build_log[0]) +
                 "\"\nPlease contact a developer and/or file a bug report.");
  }
  // get kernel
  for (const auto &kernel_name : all_kernels) {
    cl_kernel curr =
        clCreateKernel(prog, kernel_name.second.c_str(), &err_code);
    if (err_code != CL_SUCCESS)
      flogging(F_ERROR, "kernel compilation failed!" + std::to_string(err_code));
    OCLCompilerThread::eager_cache.insert({kernel_name.first, curr});
    if (kernel_name.first == hash) {
      kernel = curr;
    }
  }
  if (!kernel)
    flogging(F_ERROR,
             "something went horrible wrong for operation: " +
                 std::string(fop_to_string[node->operation->op_type]) +
                 " result type: " + std::to_string(node->operation->data_type));
  OCLCompilerThread::eager_programs.push_back(prog);
  const std::chrono::duration<double, std::milli> elapsed =
      std::chrono::high_resolution_clock::now() - start;
  flogging(F_DEBUG,
           "Compilation took " + std::to_string(elapsed.count()) + "ms");
  return kernel;
}
FGraphNode *fExecuteGraph_gpu_eagerly(FGraphNode *node) {
  if (node->result_data)
    return node;
  if (node->operation->op_type == FSTORE) {
    node->result_data = new FResultData();
    FStore *store = (FStore *)node->operation->additional_data;
    node->result_data->num_entries = store->num_entries;
    node->result_data->mem_id = store->mem_id;
    node->result_data->data = store->data;
    return node;
  }
  if (node->operation->op_type == FLATTEN ||
      node->operation->op_type == FRESHAPE) {
    // just copy previous data
    const FGraphNode *prev = node->predecessors[0];
    void const *data;
    cl_mem gpu_data;
    size_t num_elems;
    if (prev->result_data) {
      data = prev->result_data->data;
      gpu_data = prev->result_data->mem_id;
      num_elems = prev->result_data->num_entries;
    } else {
      const FStore *store = (FStore *)prev->operation->additional_data;
      data = store->data;
      gpu_data = store->mem_id;
      num_elems = store->num_entries;
    }
    FResultData *rd = new FResultData();
    rd->data = nullptr;
    rd->num_entries = num_elems;
    rd->mem_id = nullptr;
    int type_size = typeSize(node->operation->data_type);
    if (data) {
      rd->data = malloc(type_size * num_elems);
      if (!rd->data)
        flogging(F_ERROR, "Not enough memory to store result!");
      memcpy(rd->data, data, type_size * num_elems);
    } else if (gpu_data) {
      rd->mem_id = create_gpu_memory(node, CL_MEM_READ_ONLY);
      clEnqueueCopyBuffer(clqueue, gpu_data, rd->mem_id, 0, 0,
                          type_size * num_elems, 0, nullptr, nullptr);
    }
    node->result_data = rd;
    return node;
  }
  std::vector<FType> params_types(node->num_predecessor);
  for (int i = 0; i < node->num_predecessor; i++)
    params_types[i] = node->predecessors[i]->operation->data_type;
  // because the operation type should be at the same position
  int hash = OCLCompilerThread::generateKernelHash(
      node->operation->op_type, node->operation->data_type, params_types);
  const auto prog = OCLCompilerThread::eager_cache.find(hash);
  cl_kernel kernel = nullptr;
  cl_int err_code;
  std::list<cl_mem> to_free;
  if (clFinish(clqueue) != CL_SUCCESS)
    flogging(F_ERROR, "OpenCL queue error!");
  // check if the kernel already exists or if it has to be generated
  if (prog == OCLCompilerThread::eager_cache.end()) {
    kernel = OCLCompilerThread::eager_compile(node, hash);
  } else {
    kernel = prog->second;
    flogging(F_DEBUG, "Loaded existing eager kernel");
  }
  // result buffer
  size_t total_size_node;
  cl_mem res_mem = create_gpu_memory(node, CL_MEM_READ_WRITE, &total_size_node);
  node->result_data = new FResultData();
  node->result_data->mem_id = res_mem;
  node->result_data->num_entries = total_size_node;
  node->result_data->data = nullptr;
  // load parameters
  std::vector<cl_event> write_events;
  int par_index = 0;
  if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&res_mem) !=
      CL_SUCCESS)
    flogging(F_ERROR, "Could not load Argument to kernel!");
  // push operation information on demand
  switch (node->operation->op_type) {
  case FMATMUL: {
    long total_size_puffer = total_size_node;
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&total_size_puffer) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    const FGraphNode *gnp1 = node->predecessors[0],
                     *gnp2 = node->predecessors[1];
    long l = gnp1->operation->shape[gnp1->operation->dimensions - 2];
    long m = gnp1->operation->shape[gnp1->operation->dimensions - 1];
    long n = gnp2->operation->shape[gnp2->operation->dimensions - 1];
    for (long *mmd : {&l, &m, &n}) {
      if (clSetKernelArg(kernel, par_index++, sizeof(long), (void *)mmd) !=
          CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
    }
  } break;
  case FGEN_RANDOM:
  case FGRADIENT_CONVOLVE:
  case FSLIDE:
  case FCONVOLVE: {
    long total_size_puffer = total_size_node;
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&total_size_puffer) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
  } break;
  case FREDUCE_MUL:
  case FREDUCE_SUM: {
    int *dim = ((int *)node->operation->additional_data);
    if (clSetKernelArg(kernel, par_index++, sizeof(int), (void *)dim) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
  } break;
  case FEXTEND:
  case FREPEAT:
  case FSLICE: {
    long total_size_puffer = total_size_node;
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&total_size_puffer) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
  }
  default:
    break;
  }
  // process parameters i.e. predecessors
  for (int i = 0; i < node->num_predecessor; i++) {
    FGraphNode *pred = node->predecessors[i];
    FOperation *op = pred->operation;
    cl_mem mem_obj = nullptr;
    bool do_write = false;
    size_t type_size = typeSize(op->data_type);
    size_t total_size;
    cl_mem mem_id;
    if (op->op_type == FSTORE) {
      total_size = ((FStore *)op->additional_data)->num_entries;
      mem_id = ((FStore *)op->additional_data)->mem_id;
    } else {
      total_size = pred->result_data->num_entries;
      mem_id = pred->result_data->mem_id;
    }
    if (mem_id) {
      mem_obj = mem_id;
    } else {
      mem_obj = create_gpu_memory(pred, CL_MEM_READ_ONLY, &total_size);
      if (op->op_type == FSTORE) {
        ((FStore *)op->additional_data)->mem_id = mem_obj;
        if (pred->result_data)
          pred->result_data->mem_id = mem_obj;
      } else {
        pred->result_data->mem_id = mem_obj;
      }
      do_write = true;
    }
    if (do_write) {
      void *data = op->op_type == FSTORE ? ((FStore *)op->additional_data)->data
                                         : pred->result_data->data;
      err_code = clEnqueueWriteBuffer(clqueue, mem_obj, CL_TRUE, 0,
                                      total_size * type_size, data, 0, nullptr,
                                      nullptr);
      if (err_code != CL_SUCCESS) {
        std::string msg = "Unknown Error while loading data to GPU! Error: ";
        if (err_code == CL_OUT_OF_HOST_MEMORY)
          msg = "Not enough memory to load data to GPU!";
        flogging(F_ERROR, msg + std::to_string(err_code));
      }
    }
    if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&mem_obj) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    // push total element size
    if (clSetKernelArg(kernel, par_index++, sizeof(long),
                       (void *)&total_size) != CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
    // push dimensions on demand
    switch (node->operation->op_type) {
    case FMATMUL:
      if (clSetKernelArg(kernel, par_index++, sizeof(int),
                         (void *)&op->dimensions) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      break;
    case FGRADIENT_CONVOLVE: {
      if (clSetKernelArg(kernel, par_index++, sizeof(int),
                         (void *)&op->dimensions) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
    } break;
    case FSLIDE:
    case FCONVOLVE: {
      if (clSetKernelArg(kernel, par_index++, sizeof(int),
                         (void *)&op->dimensions) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
    } break;
    case FREDUCE_SUM:
    case FREDUCE_MUL: {
      int dim = ((int *)node->operation->additional_data)[0];
      const FOperation *pred = node->predecessors[0]->operation;
      long it_dim = 1; // iteration size <=> product of all dimensions along dim
      for (size_t d = dim + 1; d < pred->dimensions; d++)
        it_dim *= pred->shape[d];
      const long shape_dim = pred->shape[dim];
      if (clSetKernelArg(kernel, par_index++, sizeof(int),
                         (void *)&op->dimensions) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      if (clSetKernelArg(kernel, par_index++, sizeof(long), (void *)&it_dim) !=
          CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      if (clSetKernelArg(kernel, par_index++, sizeof(long),
                         (void *)&shape_dim) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
    } break;
    case FTRANSPOSE: {
      if (clSetKernelArg(kernel, par_index++, sizeof(int),
                         (void *)&op->dimensions) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      std::vector<long> acc_sizes_d(op->dimensions);
      std::vector<long> acc_sizes_s(op->dimensions);
      acc_sizes_d[op->dimensions - 1] = 1;
      acc_sizes_s[op->dimensions - 1] = 1;
      for (int dim = op->dimensions - 2; dim >= 0; dim--) {
        acc_sizes_d[dim] =
            acc_sizes_d[dim + 1] * node->operation->shape[dim + 1];
        acc_sizes_s[dim] = acc_sizes_s[dim + 1] * op->shape[dim + 1];
      }
      const int *transpositions = (int *)node->operation->additional_data;
      std::vector<long> acc_sizes_st(op->dimensions);
      for (int i = 0; i < op->dimensions; i++) {
        acc_sizes_st[i] = acc_sizes_s[transpositions[i]];
      }
      cl_mem asd_mem = clCreateBuffer(
          context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          op->dimensions * sizeof(long), acc_sizes_d.data(), &err_code);
      if (!asd_mem)
        flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                              std::to_string(err_code));
      if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                         (void *)&asd_mem) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      cl_mem ass_mem = clCreateBuffer(
          context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          op->dimensions * sizeof(long), acc_sizes_st.data(), &err_code);
      if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                         (void *)&ass_mem) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      if (!ass_mem)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      to_free.push_back(asd_mem);
      to_free.push_back(ass_mem);
    } break;
    case FSLICE: {
      if (clSetKernelArg(kernel, par_index++, sizeof(int),
                         (void *)&op->dimensions) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      FSlice *slice = (FSlice *)node->operation->additional_data;
      // flattened shape data
      std::vector<size_t> acc_sizes(node->operation->dimensions);
      std::vector<size_t> acc_sizes_pred(acc_sizes.size());
      for (long d = node->operation->dimensions - 1; d >= 0; d--) {
        if (d == node->operation->dimensions - 1) {
          acc_sizes[d] = 1;
          acc_sizes_pred[d] = 1;
        } else {
          acc_sizes_pred[d] = acc_sizes_pred[d + 1] * op->shape[d + 1];
          acc_sizes[d] = acc_sizes[d + 1] * node->operation->shape[d + 1];
        }
      }
      cl_mem acc_mem = clCreateBuffer(
          context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          op->dimensions * sizeof(long), acc_sizes.data(), &err_code);
      if (!acc_mem)
        flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                              std::to_string(err_code));
      cl_mem acc_pred_mem = clCreateBuffer(
          context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          op->dimensions * sizeof(long), acc_sizes_pred.data(), &err_code);
      if (!acc_pred_mem)
        flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                              std::to_string(err_code));
      // allocate steps
      cl_mem steps =
          clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         op->dimensions * sizeof(long), slice->step, &err_code);
      if (!steps)
        flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                              std::to_string(err_code));
      // calculate start and step size in flattened array
      long start = 0;
      for (unsigned int d = 0; d < node->operation->dimensions; d++) {
        start += slice->start[d] * acc_sizes_pred[d];
      }
      if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                         (void *)&acc_mem) != CL_SUCCESS ||
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
                         (void *)&op->dimensions) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      std::vector<long> acc_sizes_d(op->dimensions);
      std::vector<long> acc_sizes_s(op->dimensions);
      acc_sizes_d[op->dimensions - 1] = 1;
      acc_sizes_s[op->dimensions - 1] = 1;
      for (int dim = op->dimensions - 2; dim >= 0; dim--) {
        acc_sizes_d[dim] =
            acc_sizes_d[dim + 1] * node->operation->shape[dim + 1];
        acc_sizes_s[dim] = acc_sizes_s[dim + 1] * op->shape[dim + 1];
      }
      cl_mem asd_mem = clCreateBuffer(
          context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          op->dimensions * sizeof(long), acc_sizes_d.data(), &err_code);
      if (!asd_mem)
        flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                              std::to_string(err_code));
      if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                         (void *)&asd_mem) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      cl_mem ass_mem = clCreateBuffer(
          context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          op->dimensions * sizeof(long), acc_sizes_s.data(), &err_code);
      if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                         (void *)&ass_mem) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      if (!ass_mem)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      cl_mem predshape_mem =
          clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         op->dimensions * sizeof(long), op->shape, &err_code);
      if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                         (void *)&predshape_mem) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      to_free.push_back(asd_mem);
      to_free.push_back(ass_mem);
      to_free.push_back(predshape_mem);
    } break;
    case FEXTEND: {
      if (clSetKernelArg(kernel, par_index++, sizeof(int),
                         (void *)&op->dimensions) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      std::vector<long> acc_sizes_d(op->dimensions);
      std::vector<long> acc_sizes_s(op->dimensions);
      acc_sizes_d[op->dimensions - 1] = 1;
      acc_sizes_s[op->dimensions - 1] = 1;
      for (int dim = op->dimensions - 2; dim >= 0; dim--) {
        acc_sizes_d[dim] =
            acc_sizes_d[dim + 1] * node->operation->shape[dim + 1];
        acc_sizes_s[dim] = acc_sizes_s[dim + 1] * op->shape[dim + 1];
      }
      cl_mem asd_mem = clCreateBuffer(
          context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          op->dimensions * sizeof(long), acc_sizes_d.data(), &err_code);
      if (!asd_mem)
        flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                              std::to_string(err_code));
      if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                         (void *)&asd_mem) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      cl_mem ass_mem = clCreateBuffer(
          context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          op->dimensions * sizeof(long), acc_sizes_s.data(), &err_code);
      if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                         (void *)&ass_mem) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      if (!ass_mem)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      const FExtend *extend = (FExtend *)node->operation->additional_data;
      cl_mem steps_mem = clCreateBuffer(
          context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          op->dimensions * sizeof(long), extend->step, &err_code);
      if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                         (void *)&steps_mem) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      cl_mem start_mem = clCreateBuffer(
          context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          op->dimensions * sizeof(long), extend->start, &err_code);
      if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
                         (void *)&start_mem) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
      cl_mem predshape_mem =
          clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         op->dimensions * sizeof(long), op->shape, &err_code);
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
  // parameters for functions that dont set them per parent
  switch (node->operation->op_type) {
  case FGEN_RANDOM: {
    // push time parameter
    std::chrono::duration<double, std::nano> tm =
        std::chrono::high_resolution_clock::now().time_since_epoch();
    double t = ((unsigned long)tm.count() % 1000000) / 100.0;
    if (clSetKernelArg(kernel, par_index++, sizeof(double), (void *)&t) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
  } break;
  case FGRADIENT_CONVOLVE:
  case FSLIDE: {
    bool is_slide = node->operation->op_type == FSLIDE;
    const FOperation *op = node->operation;
    const FGraphNode *gnp1 = node->predecessors[0],
                     *gnp2 = node->predecessors[1];
    const FOperation *pred;
    const FOperation *kernel_par;
    FOperation *adjoint = nullptr;
    if (is_slide) {
      pred = gnp1->operation;
      kernel_par = gnp2->operation;
    } else {
      pred = op;
      kernel_par = gnp1->operation;
      adjoint = gnp2->operation;
      // dimensions0
      if (clSetKernelArg(kernel, par_index++, sizeof(int),
                         (void *)&node->operation->dimensions) != CL_SUCCESS)
        flogging(F_ERROR, "Could not load Argument to kernel!");
    }
    unsigned int *steps = (unsigned int *)op->additional_data;
    // calculate accumulated sizes for result, kernel and source (pred)
    std::vector<size_t> acc_sizes_pred(pred->dimensions);
    std::vector<size_t> acc_sizes_kernel(kernel_par->dimensions);
    acc_sizes_pred[pred->dimensions - 1] = 1;
    acc_sizes_kernel[kernel_par->dimensions - 1] = 1;
    for (long d = pred->dimensions - 2; d >= 0; d--) {
      acc_sizes_pred[d] = acc_sizes_pred[d + 1] * pred->shape[d + 1];
      acc_sizes_kernel[d] = acc_sizes_kernel[d + 1] * kernel_par->shape[d + 1];
    }
    cl_mem acc_pred_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        pred->dimensions * sizeof(long), acc_sizes_pred.data(), &err_code);
    if (!acc_pred_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    cl_mem acc_kernel_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       kernel_par->dimensions * sizeof(long),
                       acc_sizes_kernel.data(), &err_code);
    if (!acc_kernel_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    // allocate steps
    cl_mem steps_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       op->dimensions * sizeof(int), steps, &err_code);
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
      std::vector<size_t> acc_sizes(pred->dimensions - 1);
      acc_sizes[op->dimensions - 2] = 1;
      for (long d = op->dimensions - 3; d >= 0; d--)
        acc_sizes[d] = acc_sizes[d + 1] * adjoint->shape[d + 1];

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
    const FOperation *shape_par = is_slide ? pred : kernel_par;
    cl_mem shape_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        shape_par->dimensions * sizeof(long), shape_par->shape, &err_code);
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
    const FOperation *op = node->operation;
    const FGraphNode *gnp1 = node->predecessors[0],
                     *gnp2 = node->predecessors[1];
    const FOperation *pred = gnp1->operation, *kernel_par = gnp2->operation;
    unsigned int *steps = (unsigned int *)op->additional_data;
    // calculate accumulated sizes for result, kernel and source (pred)
    std::vector<size_t> acc_sizes(op->dimensions);
    std::vector<size_t> acc_sizes_pred(acc_sizes.size() + 1);
    std::vector<size_t> acc_sizes_kernel(acc_sizes.size() + 1);
    acc_sizes[op->dimensions - 1] = 1;
    for (long d = op->dimensions - 2; d >= 0; d--) {
      acc_sizes[d] = acc_sizes[d + 1] * op->shape[d + 1];
    }
    acc_sizes_kernel[acc_sizes.size()] = 1;
    acc_sizes_pred[acc_sizes.size()] = 1;
    for (long d = acc_sizes.size() - 1; d >= 0; d--) {
      acc_sizes_kernel[d] = acc_sizes_kernel[d + 1] * kernel_par->shape[d + 1];
      acc_sizes_pred[d] = acc_sizes_pred[d + 1] * pred->shape[d + 1];
    }
    cl_mem acc_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        op->dimensions * sizeof(long), acc_sizes.data(), &err_code);
    if (!acc_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    cl_mem acc_pred_mem = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        pred->dimensions * sizeof(long), acc_sizes_pred.data(), &err_code);
    if (!acc_pred_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    cl_mem acc_kernel_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       kernel_par->dimensions * sizeof(long),
                       acc_sizes_kernel.data(), &err_code);
    if (!acc_kernel_mem)
      flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
                            std::to_string(err_code));
    // allocate steps
    cl_mem steps_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       op->dimensions * sizeof(int), steps, &err_code);
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
  // execute it
  err_code = clEnqueueNDRangeKernel(
      clqueue, kernel, 1, nullptr, &total_size_node, nullptr,
      write_events.size(), write_events.data(), nullptr);
  for (cl_event ev : write_events)
    clReleaseEvent(ev);
  if (err_code != CL_SUCCESS) {
    std::string msg;
    switch (err_code) {
    case CL_OUT_OF_HOST_MEMORY:
      msg = "Not enough memory to execute kernel!";
      break;
    case CL_OUT_OF_RESOURCES:
      msg = "Out of resources!";
      break;
    default:
      msg = "Unknown Error during kernel execution! code: " +
            std::to_string(err_code);
      break;
    }
    flogging(F_ERROR, msg);
  }
  for (cl_mem tfn : to_free)
    clReleaseMemObject(tfn);
  return node;
}
cl_kernel OCLCompilerThread::lazy_compile(FGraphNode *node, std::string code) {
  using namespace std;
  cl_kernel kernel;
  cl_int err_code;
  // create program
  const char *code_data = code.data();
  const size_t code_length = code.length();
  cl_program prog = clCreateProgramWithSource(context, 1, &code_data,
                                              &code_length, &err_code);
  if (err_code == CL_OUT_OF_RESOURCES)
    flogging(F_ERROR, "Out of resources while creating program!");
  if (err_code == CL_OUT_OF_HOST_MEMORY)
    flogging(F_ERROR, "Not enough memory to create program!");
  // build program
  err_code = clBuildProgram(prog, 1, &device, clCompilerOpts, nullptr, nullptr);
  if (err_code == CL_INVALID_PROGRAM)
    flogging(F_ERROR,
             "Invalid Program was generated! Generated code: \"\n" + code +
                 "\"\nPlease contact a developer and/or file a bug report.");
  else if (err_code == CL_COMPILER_NOT_AVAILABLE)
    flogging(F_ERROR, "Compiler of your GPU driver is not available!");
  else if (err_code == CL_OUT_OF_HOST_MEMORY)
    flogging(F_ERROR, "Not enough memory to build program!");
  else if (err_code != CL_SUCCESS) {
    char build_log[4096];
    size_t actual_size = 0;
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 4096,
                          (void *)&build_log[0], &actual_size);
    flogging(F_ERROR,
             "Unknown Error during program compilation! Generated code: \"\n" +
                 code + "\nBuild Log:\n" + string(&build_log[0]) +
                 "\"\nPlease contact a developer and/or file a bug report.");
  }
  // get kernel
  kernel = clCreateKernel(prog, "execute_graph", &err_code);
  if (err_code != CL_SUCCESS)
    flogging(F_ERROR, "kernel compilation failed (lazy)! " + std::to_string(err_code));
  OCLCompilerThread::kernel_cache.insert({code, {prog, kernel}});
  return kernel;
}
FResultData *fSyncMemory(FGraphNode* node) {
  if (node->result_data && node->result_data->data)
    return node->result_data;
  if (node->operation->op_type == FSTORE) {
    node->result_data = new FResultData();
    FStore *store = (FStore *)node->operation->additional_data;
    node->result_data->num_entries = store->num_entries;
    node->result_data->mem_id = store->mem_id;
    node->result_data->data = store->data;
  }
  FResultData* res = node->result_data;
  if (res && res->mem_id && !res->data) {
    // read result to cpu
    int type_size_node = typeSize(node->operation->data_type);
    node->result_data->data = malloc(res->num_entries * type_size_node);
    node->result_data->num_entries = res->num_entries;
    if (!node->result_data->data)
      flogging(F_ERROR, "Not enough memory to store result!");
    // wait for result
    cl_int err_code = clEnqueueReadBuffer(
        clqueue, node->result_data->mem_id, CL_TRUE, 0,
        res->num_entries * type_size_node, res->data, 0, nullptr, nullptr);
    if (err_code != CL_SUCCESS) {
      std::string msg =
          "Unknown Error while reading the result! Error Code: " +
          std::to_string(err_code);
      if (err_code == CL_OUT_OF_HOST_MEMORY)
        msg = "Not enough memory to read result!";
      flogging(F_ERROR, msg);
    }
  }
  return res;
}
FGraphNode *fExecuteGraph_gpu(FGraphNode *node) {
  if (!initialized) {
    flintInit_gpu();
  }
  {
    if (node->operation->op_type == FSTORE) {
      node->result_data = new FResultData();
      FStore *store = (FStore *)node->operation->additional_data;
      node->result_data->num_entries = store->num_entries;
      node->result_data->mem_id = store->mem_id;
      node->result_data->data = store->data;
    }
    if (node->result_data)
      return node;
  }
  auto start = std::chrono::high_resolution_clock::now();
  FResultData *resultData = new FResultData();
  FOperation *node_op = node->operation;
  size_t total_size_node = 1;
  for (int i = 0; i < node_op->dimensions; i++)
    total_size_node *= node_op->shape[i];
  // calculate Code and Parameters
  using namespace std;
  list<pair<FGraphNode *, string>> parameters;
  unordered_set<string> additional_params;
  string graph_code = generateCode(node, parameters, additional_params);
  string code = "__kernel void execute_graph(__global ";
  code += typeString(node->operation->data_type);
  code += " *R";
  // insert parameters
  for (auto &[op, name] : parameters)
    code += ", __global const " + typeString(op->operation->data_type) + " *" +
            name;
  if (additional_params.contains("time"))
    code += ", const double time";
  code += "){\n";
  // add the execution code
  code += graph_code;
  // store result
  code += "R[index] = v0;\n}";
  chrono::duration<double, std::milli> elapsed =
      chrono::high_resolution_clock::now() - start;
  flogging(F_DEBUG, "code generation finished (in " +
                        to_string(elapsed.count()) + " ms): \n" + code);
  // don't create code when in cache
  auto cache_val = OCLCompilerThread::kernel_cache.find(code);
  cl_kernel kernel = nullptr;
  cl_int err_code;
  if (cache_val == OCLCompilerThread::kernel_cache.end()) {
    kernel = OCLCompilerThread::lazy_compile(node, code);
  } else {
    flogging(F_DEBUG, "code from cache");
    kernel = cache_val->second.second;
  }
  chrono::duration<double, std::milli> compilation_time =
      chrono::high_resolution_clock::now() - start;
  start = std::chrono::high_resolution_clock::now();
  // result buffer
  size_t type_size_node = typeSize(node_op->data_type);
  cl_mem result_mem =
      clCreateBuffer(context, CL_MEM_READ_WRITE,
                     total_size_node * type_size_node, nullptr, &err_code);
  resultData->mem_id = result_mem;
  if (err_code == CL_OUT_OF_HOST_MEMORY)
    flogging(F_ERROR, "Not enough memory to create buffer!");
  int index = 1;
  std::vector<cl_event> writeEvents;
  for (auto &[gn, name] : parameters) {
    FOperation *op = gn->operation;
    // TODO keep track of when data in Store is changed
    cl_mem mem_obj = nullptr;
    bool do_write = false;
    size_t type_size = typeSize(op->data_type);
    size_t total_size = op->op_type == FSTORE
                            ? ((FStore *)op->additional_data)->num_entries
                            : gn->result_data->num_entries;
    cl_mem mem_id = op->op_type == FSTORE
                        ? ((FStore *)op->additional_data)->mem_id
                        : gn->result_data->mem_id;
    if (mem_id) {
      mem_obj = mem_id;
    } else {
      mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                               total_size * type_size, nullptr, &err_code);
      if (err_code == CL_OUT_OF_HOST_MEMORY)
        flogging(F_ERROR, "Not enough memory to create buffer!");
      if (op->op_type == FSTORE)
        ((FStore *)op->additional_data)->mem_id = mem_obj;
      if (gn->result_data)
        gn->result_data->mem_id = mem_obj;
      do_write = true;
    }
    // actually write the buffer
    if (do_write) {
      void *data = op->op_type == FSTORE ? ((FStore *)op->additional_data)->data
                                         : gn->result_data->data;
      writeEvents.emplace_back();
      err_code = clEnqueueWriteBuffer(clqueue, mem_obj, CL_FALSE, 0,
                                      total_size * type_size, data, 0, nullptr,
                                      &writeEvents[writeEvents.size() - 1]);
      if (err_code != CL_SUCCESS) {
        string msg = "Unknown Error while loading data to GPU!";
        if (err_code == CL_OUT_OF_HOST_MEMORY)
          msg = "Not enough memory to load data to GPU!";
        flogging(F_ERROR, msg);
      }
    }
    if (clSetKernelArg(kernel, index++, sizeof(cl_mem), (void *)&mem_obj) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
  }
  if (clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&result_mem) !=
      CL_SUCCESS)
    flogging(F_ERROR, "Could not set Kernel Argument for the result!");
  // some operations need additional parameters
  if (additional_params.contains("time")) {
    std::chrono::duration<double, std::nano> tm =
        std::chrono::high_resolution_clock::now().time_since_epoch();
    double t = ((unsigned long)tm.count() % 1000000) / 100.0;

    if (clSetKernelArg(kernel, index++, sizeof(double), (void *)&t) !=
        CL_SUCCESS)
      flogging(F_ERROR, "Could not load Argument to kernel!");
  }
  // execute kernel
  const size_t global_size = total_size_node;

  err_code =
      clEnqueueNDRangeKernel(clqueue, kernel, 1, nullptr, &global_size, nullptr,
                             writeEvents.size(), writeEvents.data(), nullptr);
  for (cl_event ev : writeEvents)
    clReleaseEvent(ev);
  if (err_code != CL_SUCCESS) {
    string msg;
    switch (err_code) {
    case CL_OUT_OF_HOST_MEMORY:
      msg = "Not enough memory to execute kernel!";
      break;
    case CL_OUT_OF_RESOURCES:
      msg = "Out of resources!";
      break;
    default:
      msg = "Unknown Error during kernel execution!";
      break;
    }
    flogging(F_ERROR, msg);
  }
  resultData->num_entries = total_size_node;
  elapsed = chrono::high_resolution_clock::now() - start;
  flogging(F_DEBUG, "compilation took " + to_string(compilation_time.count()) +
                        "ms, execution took " + to_string(elapsed.count()));
  node->result_data = resultData;
  return node;
}
void flintCleanup_gpu() {
  if (initialized) {
    flogging(F_DEBUG, "Cleaning up GPU Backend");
    initialized = false;
    for (auto &k : OCLCompilerThread::kernel_cache) {
      clReleaseKernel(k.second.second);
      clReleaseProgram(k.second.first);
    }
    for (auto &k : OCLCompilerThread::eager_cache) {
      clReleaseKernel(k.second);
    }
    for (auto &p : OCLCompilerThread::eager_programs)
      clReleaseProgram(p);
    OCLCompilerThread::kernel_cache.clear();
    OCLCompilerThread::eager_cache.clear();
    OCLCompilerThread::eager_programs.clear();
    clReleaseCommandQueue(clqueue);
    clReleaseDevice(device);
    clReleaseContext(context);
  }
}
