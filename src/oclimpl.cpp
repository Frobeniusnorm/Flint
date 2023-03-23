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
#include "utils.hpp"
#include <CL/cl.h>
#include <list>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <tuple>
#include <typeinfo>
#include <unordered_map>
#include <vector>
static void openclCallback(const char *errinfo, const void *privateinfo,
                           size_t cb, void *user_data) {
  flogging(F_WARNING, "{OpenCL} " + std::string(errinfo));
}

static bool initialized = false;
// opencl vars
static cl_context context;
static cl_command_queue queue;
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
  queue = clCreateCommandQueueWithProperties(context, device, NULL, &status);
  if (status != CL_SUCCESS)
    flogging(F_ERROR, "clCreateCommandQueue");
  initialized = true;
  flogging(F_VERBOSE, "Flint GPU backend was initialized!");
}

#include <chrono>
#include <unordered_map>
static std::unordered_map<std::string, std::pair<cl_program, cl_kernel>>
    kernel_cache;
FGraphNode *fExecuteGraph_gpu_eagerly(FGraphNode *node) {
  // maybe there are still major optimizations we can do here
  return fExecuteGraph_gpu(node);
}
FGraphNode *fExecuteGraph_gpu(FGraphNode *node) {
  if (!initialized) {
    flintInit_gpu();
  }
  if (node->result_data)
    return node;
  if (node->operation->op_type == FCONST) {
    node->result_data = new FResultData();
    node->result_data->num_entries = 1;
    node->result_data->mem_id = nullptr;
    node->result_data->data =
        ((FConst *)node->operation->additional_data)->value;
    return node;
  }
  if (node->operation->op_type == FSTORE) {
    node->result_data = new FResultData();
    FStore *store = (FStore *)node->operation->additional_data;
    node->result_data->num_entries = store->num_entries;
    node->result_data->mem_id = store->mem_id;
    node->result_data->data = store->data;
    return node;
  }
  // ensures all previous operations are finished
  if (clFinish(queue) != CL_SUCCESS) {
    flogging(F_ERROR, "OpenCL queue error!");
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
  string graph_code = generateCode(node, parameters);
  string code = "__kernel void execute_graph(__global ";
  code += typeString(node->operation->data_type);
  code += " *R";
  // insert parameters
  for (auto &[op, name] : parameters)
    code += ", __global const " + typeString(op->operation->data_type) + " *" +
            name;
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
  auto cache_val = kernel_cache.find(code);
  cl_kernel kernel = nullptr;
  cl_int err_code;
  if (cache_val == kernel_cache.end()) {
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
    err_code = clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);
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
      flogging(
          F_ERROR,
          "Unknown Error during program compilation! Generated code: \"\n" +
              code + "\nBuild Log:\n" + string(&build_log[0]) +
              "\"\nPlease contact a developer and/or file a bug report.");
    }
    // get kernel
    kernel = clCreateKernel(prog, "execute_graph", &err_code);
    if (err_code != CL_SUCCESS)
      flogging(F_ERROR, "kernel compilation failed!");
    kernel_cache.insert({code, {prog, kernel}});
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
    bool doWrite = false;
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
      else
        gn->result_data->mem_id = mem_obj;
      doWrite = true;
    }
    // actually write the buffer
    if (doWrite) {
      void *data = op->op_type == FSTORE ? ((FStore *)op->additional_data)->data
                                         : gn->result_data->data;
      writeEvents.emplace_back();
      err_code = clEnqueueWriteBuffer(queue, mem_obj, CL_TRUE, 0,
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
  // execute kernel
  const size_t global_size = total_size_node;
  err_code =
      clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr,
                             writeEvents.size(), writeEvents.data(), nullptr);
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
  resultData->data = malloc(total_size_node * type_size_node);
  resultData->num_entries = total_size_node;
  if (!resultData->data)
    flogging(F_ERROR, "Not enough memory to store result!");
  // wait for result
  err_code = clEnqueueReadBuffer(queue, result_mem, CL_TRUE, 0,
                                 total_size_node * type_size_node,
                                 (void *)resultData->data, 0, nullptr, nullptr);
  if (err_code != CL_SUCCESS) {
    string msg = "Unknown Error while reading the result!";
    if (err_code == CL_OUT_OF_HOST_MEMORY)
      msg = "Not enough memory to read result!";
    flogging(F_ERROR, msg);
  }
  elapsed = chrono::high_resolution_clock::now() - start;
  flogging(F_DEBUG, "compilation took " + to_string(compilation_time.count()) +
                        "ms, execution took " + to_string(elapsed.count()));
  node->result_data = resultData;
  return node;
}
void flintCleanup_gpu() {
  if (initialized) {
    initialized = false;
    clReleaseDevice(device);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    for (auto &k : kernel_cache) {
      clReleaseKernel(k.second.second);
      clReleaseProgram(k.second.first);
    }
  }
}
