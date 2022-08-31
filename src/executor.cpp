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

  This file includes the implementation of the GPU backend and the backend
  selector function.
*/

#include "../flint.h"
#include "logger.hpp"
#include "utils.hpp"
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
  log(WARNING, "{OpenCL} " + std::string(errinfo));
}
void flintInit(int cpu, int gpu) {
  log(VERBOSE, "Initializing Flint");
  if (cpu)
    flintInit_cpu();
  if (gpu)
    flintInit_gpu();
}
static bool initialized = false;
// opencl vars
static cl_context context;
static cl_command_queue queue;
static cl_device_id device;
void flintInit_gpu() {
  cl_platform_id platform = NULL;
  device = NULL;
  cl_uint num_dev, num_plat;
  if (clGetPlatformIDs(1, &platform, &num_plat) != CL_SUCCESS)
    log(ERROR, "clGetPlatformIds");
  if (num_plat == 0)
    log(ERROR, "Could not find any OpenCL Platform available! Please make "
               "sure, you have setup your OpenCL driver right!");
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, &num_dev) !=
      CL_SUCCESS)
    log(ERROR, "clGetDeviceIds");
  if (num_dev == 0)
    log(ERROR, "Could not find any OpenCL devices available! Please make sure, "
               "you have setup your OpenCL driver right!");
  char dev_name[128];
  size_t dev_name_size;
  char dev_vers[128];
  size_t dev_vers_size;
  char dev_vend[128];
  size_t dev_vend_size;
  cl_device_type dev_type;
  size_t dev_type_size;
  clGetDeviceInfo(device, CL_DEVICE_NAME, 128 * sizeof(char),
                  (void *)&dev_name[0], &dev_name_size);
  clGetDeviceInfo(device, CL_DEVICE_VERSION, 128, (void *)&dev_vers[0],
                  &dev_vers_size);
  clGetDeviceInfo(device, CL_DEVICE_VENDOR, 128, (void *)&dev_vend[0],
                  &dev_vend_size);
  clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(dev_type), (void *)&dev_type,
                  &dev_type_size);
  std::string dev_type_string;
  if ((dev_type & CL_DEVICE_TYPE_CPU) == CL_DEVICE_TYPE_CPU) {
    dev_type_string = "CPU";
  } else if ((dev_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU) {
    dev_type_string = "GPU";
  } else if ((dev_type & CL_DEVICE_TYPE_ACCELERATOR) ==
             CL_DEVICE_TYPE_ACCELERATOR) {
    dev_type_string = "Accelerator";
  } else
    dev_type_string = "Device";
  std::string info =
      "Using " + dev_type_string + " '" + std::string(dev_vend, dev_vend_size) +
      "', '" + std::string(dev_name, dev_name_size) + "' with OpenCL version " +
      std::string(dev_vers, dev_vers_size);
  log(INFO, info);
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
    log(ERROR, err);
  }
  queue = clCreateCommandQueueWithProperties(context, device, NULL, &status);
  if (status != CL_SUCCESS)
    log(ERROR, "clCreateCommandQueue");
  initialized = true;
  log(VERBOSE, "Flint GPU backend was initialized!");
}
void flintCleanup() {
  flintCleanup_cpu();
  flintCleanup_gpu();
}
void flintCleanup_gpu() {
  if (initialized) {
    initialized = false;
    clReleaseDevice(device);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
  }
}
void setLoggingLevel(int level) { setLoggerLevel(level); }

static std::string
generateCode(FGraphNode *node,
             std::list<std::pair<FOperation *, std::string>> &parameters) {
  using namespace std;
  list<pair<FGraphNode *, string>> todo;
  unordered_map<FOperation *, std::string> assigned_params;
  int variable_index = 0;
  string code = "";
  todo.push_front({node, "v0"});
  while (!todo.empty()) {
    // take from queue
    const auto [node, name] = todo.front();
    todo.pop_front();
    bool push_pred = true;
    // write code
    string type = typeString(node->operation->data_type);
    switch (node->operation->op_type) {
    case RESULTDATA:
    case STORE: {
      push_pred = false;
      if (assigned_params.find(node->operation) != assigned_params.end()) {
        code = type + " " + name + " = " + assigned_params[node->operation] +
               ";\n" + code;
      } else {
        assigned_params.insert({node->operation, name});
        parameters.push_front({node->operation, name});
      }
    } break;
    case CONST:
      switch (node->operation->data_type) {
      case INT32: {
        FConst *actcst = (FConst *)node->operation->additional_data;
        code = type + " " + name + " = " + to_string(*((int *)actcst->value)) +
               ";\n" + code;
      } break;
      case INT64: {
        FConst *actcst = (FConst *)node->operation->additional_data;
        code = type + " " + name + " = " + to_string(*((long *)actcst->value)) +
               ";\n" + code;
      } break;
      case FLOAT64: {
        FConst *actcst = (FConst *)node->operation->additional_data;
        code = type + " " + name + " = " +
               to_string(*((double *)actcst->value)) + ";\n" + code;
      } break;
      case FLOAT32: {
        FConst *actcst = (FConst *)node->operation->additional_data;
        code = type + " " + name + " = " +
               to_string(*((float *)actcst->value)) + ";\n" + code;
      } break;
      }
      break;
    // Binary Operators
    case ADD:
    case SUB:
    case DIV:
    case MUL: {
      // size of current variable has to be equal to the size of one opperand,
      // the other one is at least smaller but not larger
      char op = '\0';
      switch (node->operation->op_type) {
      case ADD:
        op = '+';
        break;
      case SUB:
        op = '-';
        break;
      case DIV:
        op = '/';
        break;
      case MUL:
        op = '*';
        break;
      }
      code = type + " " + name + " = v" + to_string(variable_index + 1) + " " +
             op + " v" + to_string(variable_index + 2) + ";\n" + code;
      break;
    }
    case POW: {
      FOperation *x = node->predecessors[0]->operation;
      FOperation *y = node->predecessors[1]->operation;
      if ((x->data_type == FLOAT32 || x->data_type == FLOAT64) &&
          (y->data_type == FLOAT32 && y->data_type == FLOAT64))
        code = type + " " + name + " = pow(v" + to_string(variable_index + 1) +
               ", v" + to_string(variable_index + 2) + ");\n" + code;
      else if (x->data_type == INT64 &&
               (y->data_type == INT32 || y->data_type == INT64))
        code = type + " " + name + " = (long)pown((double)v" +
               to_string(variable_index + 1) + ", (int)v" +
               to_string(variable_index + 2) + ");\n" + code;
      else if (x->data_type == INT32 &&
               (y->data_type == INT32 || y->data_type == INT64))
        code = type + " " + name + " = (int)pown((float)v" +
               to_string(variable_index + 1) + ", (int)v" +
               to_string(variable_index + 2) + ");\n" + code;
      else
        code = type + " " + name + " = pow((double)v" +
               to_string(variable_index + 1) + ", (double)v" +
               to_string(variable_index + 2) + ");\n" + code;
    } break; // TODO matmul
    case FLATTEN: {
      code = type + " " + name + " = v" + to_string(variable_index + 1) +
             ";\n" + code;
    } break;
    }
    // push predecessors
    if (push_pred)
      for (int i = 0; i < node->num_predecessor; i++)
        todo.push_front(
            {node->predecessors[i], "v" + to_string(++variable_index)});
  }
  return code;
}
FGraphNode *executeGraph(FGraphNode *node) {
  // TODO
  return executeGraph_gpu(node);
}
#include <chrono>
#include <unordered_map>
static std::unordered_map<std::string, std::pair<cl_program, cl_kernel>>
    kernel_cache;
FGraphNode *executeGraph_gpu(FGraphNode *node) {
  if (!initialized) {
    flintInit_gpu();
  }
  auto start = std::chrono::high_resolution_clock::now();
  FOperation *result = new FOperation();
  FResultData *resultData = new FResultData();
  result->op_type = RESULTDATA;
  result->data_type = node->operation->data_type;
  result->additional_data = resultData;
  FGraphNode *newsucc = new FGraphNode();
  newsucc->num_predecessor = 1;
  newsucc->reference_counter = 0;
  newsucc->predecessors = safe_mal<FGraphNode *>(1);
  newsucc->predecessors[0] = node;
  node->reference_counter++;
  newsucc->operation = result;
  // calculate Code and Parameters
  using namespace std;
  list<pair<FOperation *, string>> parameters;
  string graph_code = generateCode(node, parameters);
  string code = "__kernel void execute_graph(__global ";
  code += typeString(node->operation->data_type);
  code += " *R";
  // insert parameters
  int par_idx = 0;
  for (auto &[op, name] : parameters)
    code += ", __global const " + typeString(op->data_type) + " *P" +
            to_string(par_idx++);
  code += "){\n";
  // bind parameters to variables
  // currently we only work on flat arrays
  code += "int index = get_global_id(0);\n";
  par_idx = 0;
  for (auto &[op, name] : parameters) {
    string type = typeString(op->data_type);
    code += type + " " + name + " = ";
    string indx_mod = "";
    if (op->dimensions < node->operation->dimensions) {
      // we take the remainder of the division of the product of the sizes of
      // the dimensions that are not shared by op
      int factor = 1;
      // for (int i = 0; i < node->operation->dimensions - op->dimensions; i++)
      //   factor *= node->operation->shape[i];
      for (int i = 0; i < op->dimensions; i++) {
        if (node->operation
                ->shape[i + (node->operation->dimensions - op->dimensions)] !=
            op->shape[i])
          log(ERROR,
              "incompatible shapes of operands: " +
                  vectorString(vector<int>(node->operation->shape,
                                           node->operation->shape +
                                               node->operation->dimensions)) +
                  " and " +
                  vectorString(
                      vector<int>(op->shape, op->shape + op->dimensions)));
        factor *= op->shape[i];
      }
      indx_mod = "%" + to_string(factor);
    }
    code += "P" + to_string(par_idx) + "[index" + indx_mod + "];\n";
    par_idx++;
  }
  // add the execution code
  code += graph_code;
  // store result
  code += "R[index] = v0;\n}";
  chrono::duration<double, std::milli> elapsed =
      chrono::high_resolution_clock::now() - start;
  log(DEBUG, "code generation finished (in " + to_string(elapsed.count()) +
                 " ms): \n" + code);
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
      log(ERROR, "Out of resources while creating program!");
    if (err_code == CL_OUT_OF_HOST_MEMORY)
      log(ERROR, "Not enough memory to create program!");
    // build program
    err_code = clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);
    if (err_code == CL_INVALID_PROGRAM)
      log(ERROR,
          "Invalid Program was generated! Generated code: \"\n" + code +
              "\"\nPlease contact a developer and/or file a bug report.");
    else if (err_code == CL_COMPILER_NOT_AVAILABLE)
      log(ERROR, "Compiler of your GPU driver is not available!");
    else if (err_code == CL_OUT_OF_HOST_MEMORY)
      log(ERROR, "Not enough memory to build program!");
    else if (err_code != CL_SUCCESS) {
      char build_log[4096];
      size_t actual_size = 0;
      clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 4096,
                            (void *)&build_log[0], &actual_size);
      log(ERROR,
          "Unknown Error during program compilation! Generated code: \"\n" +
              code + "\nBuild Log:\n" + string(&build_log[0]) +
              "\"\nPlease contact a developer and/or file a bug report.");
    }
    // get kernel
    kernel = clCreateKernel(prog, "execute_graph", &err_code);
    if (err_code != CL_SUCCESS)
      log(ERROR, "kernel compilation failed!");
    kernel_cache.insert({code, {prog, kernel}});
  } else {
    log(DEBUG, "code from cache");
    kernel = cache_val->second.second;
  }
  chrono::duration<double, std::milli> compilation_time =
      chrono::high_resolution_clock::now() - start;
  start = std::chrono::high_resolution_clock::now();
  // result buffer
  FOperation *node_op = node->operation;
  size_t total_size_node = 1;
  for (int i = 0; i < node_op->dimensions; i++)
    total_size_node *= node_op->shape[i];
  size_t type_size_node = typeSize(node_op->data_type);
  cl_mem result_mem =
      clCreateBuffer(context, CL_MEM_READ_WRITE,
                     total_size_node * type_size_node, nullptr, &err_code);
  resultData->mem_id = result_mem;
  if (err_code == CL_OUT_OF_HOST_MEMORY)
    log(ERROR, "Not enough memory to create buffer!");
  int index = 1;
  std::vector<cl_event> writeEvents;
  for (auto &[op, name] : parameters) {
    // TODO keep track of when data in Store is changed
    cl_mem mem_obj = nullptr;
    bool doWrite =
        op->op_type ==
        STORE; // can always be changed, ResultData only changes with gpu data
    size_t type_size = typeSize(op->data_type);
    size_t total_size = op->op_type == STORE
                            ? ((FStore *)op->additional_data)->num_entries
                            : ((FResultData *)op->additional_data)->num_entries;
    cl_mem mem_id = op->op_type == STORE
                        ? ((FStore *)op->additional_data)->mem_id
                        : ((FResultData *)op->additional_data)->mem_id;
    if (mem_id) {
      mem_obj = mem_id;
    } else {
      mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                               total_size * type_size, nullptr, &err_code);
      if (err_code == CL_OUT_OF_HOST_MEMORY)
        log(ERROR, "Not enough memory to create buffer!");
      if (op->op_type == STORE)
        ((FStore *)op->additional_data)->mem_id = mem_obj;
      else
        ((FResultData *)op->additional_data)->mem_id = mem_obj;
      doWrite = true;
    }
    // actually write the buffer
    if (doWrite) {
      void *data = op->op_type == STORE
                       ? ((FStore *)op->additional_data)->data
                       : ((FResultData *)op->additional_data)->data;
      string buffer_data = "";
      for (int i = 0; i < total_size; i++)
        if (op->data_type == FLOAT32)
          buffer_data += to_string(((float *)data)[i]) + " ";
        else if (op->data_type == FLOAT64)
          buffer_data += to_string(((double *)data)[i]) + " ";
        else if (op->data_type == INT32)
          buffer_data += to_string(((int *)data)[i]) + " ";
        else
          buffer_data += to_string(((long *)data)[i]) + " ";
      writeEvents.emplace_back();
      err_code = clEnqueueWriteBuffer(queue, mem_obj, CL_TRUE, 0,
                                      total_size * type_size, data, 0, nullptr,
                                      &writeEvents[writeEvents.size() - 1]);
      if (err_code != CL_SUCCESS) {
        string msg = "Unknown Error while loading data to GPU!";
        if (err_code == CL_OUT_OF_HOST_MEMORY)
          msg = "Not enough memory to load data to GPU!";
        log(ERROR, msg);
      }
    }
    if (clSetKernelArg(kernel, index++, sizeof(cl_mem), (void *)&mem_obj) !=
        CL_SUCCESS)
      log(ERROR, "Could not load Argument to kernel!");
  }
  if (clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&result_mem) !=
      CL_SUCCESS)
    log(ERROR, "Could not set Kernel Argument for the result!");
  // execute kernel
  const size_t global_size = total_size_node;
  const size_t local_size = 1;
  err_code = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size,
                                    &local_size, writeEvents.size(),
                                    writeEvents.data(), nullptr);
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
    log(ERROR, msg);
  }
  // size for result
  result->dimensions = node_op->dimensions;
  result->shape = safe_mal<int>(result->dimensions);
  memcpy((void *)result->shape, (void *)node_op->shape,
         result->dimensions * sizeof(int));
  resultData->data = malloc(total_size_node * type_size_node);
  resultData->num_entries = total_size_node;
  if (!resultData->data)
    log(ERROR, "Not enough memory to store result!");
  // wait for result
  err_code = clEnqueueReadBuffer(queue, result_mem, CL_TRUE, 0,
                                 total_size_node * type_size_node,
                                 (void *)resultData->data, 0, nullptr, nullptr);
  if (err_code != CL_SUCCESS) {
    string msg = "Unknown Error while reading the result!";
    if (err_code == CL_OUT_OF_HOST_MEMORY)
      msg = "Not enough memory to read result!";
    log(ERROR, msg);
  }
  elapsed = chrono::high_resolution_clock::now() - start;
  log(DEBUG, "compilation took " + to_string(compilation_time.count()) +
                 "ms, execution took " + to_string(elapsed.count()) + "ms");
  return newsucc;
}
