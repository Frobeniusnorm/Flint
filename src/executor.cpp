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
  flogging(F_WARNING, "{OpenCL} " + std::string(errinfo));
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
    flogging(F_ERROR, "clGetPlatformIds");
  if (num_plat == 0)
    flogging(F_ERROR,
             "Could not find any OpenCL Platform available! Please make "
             "sure, you have setup your OpenCL driver right!");
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, &num_dev) !=
      CL_SUCCESS)
    flogging(F_ERROR, "clGetDeviceIds");
  if (num_dev == 0)
    flogging(F_ERROR,
             "Could not find any OpenCL devices available! Please make sure, "
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

void flintCleanup_gpu() {
  if (initialized) {
    initialized = false;
    clReleaseDevice(device);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
  }
}

static std::string
generateCode(FGraphNode *node,
             std::list<std::pair<FOperation *, std::string>> &parameters) {
  using namespace std;
  list<tuple<FGraphNode *, string>> todo;
  unordered_map<FOperation *, std::string> assigned_params;
  int variable_index = 0;
  string code = "";

  // indexing logic
  string index_defs = "int index = get_global_id(0);\n";
  unsigned int num_indices = 0;

  todo.push_front({node, "v0"});
  while (!todo.empty()) {
    // take from queue
    const auto [node, name] = todo.front();
    todo.pop_front();
    bool push_pred = true;
    // write code
    string type = typeString(node->operation->data_type);
    switch (node->operation->op_type) {
    case FRESULTDATA:
    case FSTORE: {
      push_pred = false;
      size_t num_entries =
          node->operation->op_type == FSTORE
              ? ((FStore *)node->operation->additional_data)->num_entries
              : ((FResultData *)node->operation->additional_data)->num_entries;
      if (assigned_params.find(node->operation) != assigned_params.end()) {
        code = type + " " + name + " = " + assigned_params[node->operation] +
               "[index%" + to_string(num_entries) + "];\n" + code;
      } else {
        size_t pid = assigned_params.size();
        assigned_params.insert({node->operation, "P" + to_string(pid)});
        parameters.push_back({node->operation, "P" + to_string(pid)});
        code = type + " " + name + " = P" + to_string(pid) + "[index%" +
               to_string(num_entries) + "];\n" + code;
      }
    } break;
    case FCONST:
      switch (node->operation->data_type) {
      case F_INT32: {
        FConst *actcst = (FConst *)node->operation->additional_data;
        code = type + " " + name + " = " + to_string(*((int *)actcst->value)) +
               ";\n" + code;
      } break;
      case F_INT64: {
        FConst *actcst = (FConst *)node->operation->additional_data;
        code = type + " " + name + " = " + to_string(*((long *)actcst->value)) +
               ";\n" + code;
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
      code = type + " " + name + " = v" + to_string(variable_index + 1) + " " +
             op + " v" + to_string(variable_index + 2) + ";\n" + code;
      break;
    }
    case FPOW: {
      FOperation *x = node->predecessors[0]->operation;
      FOperation *y = node->predecessors[1]->operation;
      if ((x->data_type == F_FLOAT32 || x->data_type == F_FLOAT64) &&
          (y->data_type == F_FLOAT32 || y->data_type == F_FLOAT64))
        code = type + " " + name + " = pow(v" + to_string(variable_index + 1) +
               ", v" + to_string(variable_index + 2) + ");\n" + code;
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
      code = type + " " + name + " = min(v" + to_string(variable_index + 1) +
             ", v" + to_string(variable_index + 2) + ");\n" + code;

    } break;
    case FMAX: {
      code = type + " " + name + " = max(v" + to_string(variable_index + 1) +
             ", v" + to_string(variable_index + 2) + ");\n" + code;

    } break;
    case FMATMUL: {
      string par1, par2;
      push_pred = false;
      FGraphNode *gnp1 = node->predecessors[0], *gnp2 = node->predecessors[1];
      // we ignore the value assignment of the parameters since we have to
      // access the arrays directly parameter 1
      if (assigned_params.find(gnp1->operation) != assigned_params.end()) {
        par1 = assigned_params[gnp1->operation];
      } else {
        par1 = "P" + to_string(assigned_params.size());
        assigned_params.insert({gnp1->operation, par1});
        parameters.push_back({gnp1->operation, par1});
      }
      // parameter 2
      if (assigned_params.find(gnp2->operation) != assigned_params.end()) {
        par2 = assigned_params[gnp2->operation];
      } else {
        par2 = "P" + to_string(assigned_params.size());
        assigned_params.insert({gnp2->operation, par2});
        parameters.push_back({gnp2->operation, par2});
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
      code = type + " " + name + " = abs(v" +
             std::to_string(variable_index + 1) + ");\n" + code;
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
      code = type + " " + name + " = log2(v" +
             std::to_string(variable_index + 1) + ");\n" + code;
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
      if (assigned_params.find(prev->operation) != assigned_params.end()) {
        par1 = assigned_params[prev->operation];
      } else {
        par1 = "P" + to_string(assigned_params.size());
        parameters.push_back({prev->operation, par1});
        assigned_params.insert({prev->operation, par1});
      }
      reduce_code +=
          " " + name +
          (node->operation->op_type == FREDUCE_SUM ? " += " : " *= ") + par1 +
          "[(index / " + std::to_string(it_dim) + ") * " +
          std::to_string(it_dim) + " * " +
          std::to_string(prev->operation->shape[red_dim]) + " + index % " +
          std::to_string(it_dim) + " + i * " + std::to_string(it_dim) +
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
      variable_index--; // because we dont generate a variable
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
      // to get the index in the source array we first calculate the indices and
      // reproject
      index_defs += "{\nint working_index = index;\nindex = 0;\n";
      for (int dim = 0; dim < op->dimensions; dim++) {
        index_defs += "index += ((working_index /" +
                      to_string(acc_sizes_d[dim]) + ") % " +
                      to_string(pred->shape[dim]) + ") * " +
                      to_string(acc_sizes_s[dim]) + ";\n";
        index_defs += "working_index %= " + to_string(acc_sizes_d[dim]) + ";\n";
      }
      index_defs += "}\n";
      code = "index = old_index" + to_string(old_idx) + ";\n" + code;
      variable_index--; // because we dont generate a variable
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
      // to get the index in the source array we first calculate the indices and
      // reproject
      index_defs += "{\nint working_index = index;\nindex = 0;\n";
      for (int dim = 0; dim < op->dimensions; dim++) {
        index_defs += "index += (working_index /" +
                      to_string(acc_sizes_d[dim]) + ") * " +
                      to_string(acc_sizes_s[transposition[dim]]) + ";\n";
        index_defs += "working_index %= " + to_string(acc_sizes_d[dim]) + ";\n";
      }
      index_defs += "}\n";
      code = "index = old_index" + to_string(old_idx) + ";\n" + code;
      variable_index--; // because we dont generate a variable
    } break;
    default:
      break;
    }
    // push predecessors dfs
    if (push_pred)
      for (int i = 0; i < node->num_predecessor; i++)
        todo.push_front(
            {node->predecessors[i], "v" + to_string(++variable_index)});
  }
  code = index_defs + code;
  return code;
}

#include <chrono>
#include <unordered_map>
static std::unordered_map<std::string, std::pair<cl_program, cl_kernel>>
    kernel_cache;
FGraphNode *fExecuteGraph_gpu_eagerly(FGraphNode *node) {
  // TODO
  return fExecuteGraph_gpu(node);
}
FGraphNode *fExecuteGraph_gpu(FGraphNode *node) {
  if (!initialized) {
    flintInit_gpu();
  }
  auto start = std::chrono::high_resolution_clock::now();
  FOperation *result = new FOperation();
  FResultData *resultData = new FResultData();
  result->op_type = FRESULTDATA;
  result->data_type = node->operation->data_type;
  result->additional_data = resultData;
  FGraphNode *newsucc = new FGraphNode();
  newsucc->num_predecessor = 1;
  newsucc->reference_counter = 0;
  newsucc->predecessors = safe_mal<FGraphNode *>(1);
  newsucc->predecessors[0] = node;
  node->reference_counter++;
  newsucc->operation = result;
  FOperation *node_op = node->operation;
  size_t total_size_node = 1;
  for (int i = 0; i < node_op->dimensions; i++)
    total_size_node *= node_op->shape[i];
  // calculate Code and Parameters
  using namespace std;
  list<pair<FOperation *, string>> parameters;
  string graph_code = generateCode(node, parameters);
  string code = "__kernel void execute_graph(__global ";
  code += typeString(node->operation->data_type);
  code += " *R";
  // insert parameters
  for (auto &[op, name] : parameters)
    code += ", __global const " + typeString(op->data_type) + " *" + name;
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
  for (auto &[op, name] : parameters) {
    // TODO keep track of when data in Store is changed
    cl_mem mem_obj = nullptr;
    bool doWrite =
        op->op_type ==
        FSTORE; // can always be changed, ResultData only changes with gpu data
    size_t type_size = typeSize(op->data_type);
    size_t total_size = op->op_type == FSTORE
                            ? ((FStore *)op->additional_data)->num_entries
                            : ((FResultData *)op->additional_data)->num_entries;
    cl_mem mem_id = op->op_type == FSTORE
                        ? ((FStore *)op->additional_data)->mem_id
                        : ((FResultData *)op->additional_data)->mem_id;
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
        ((FResultData *)op->additional_data)->mem_id = mem_obj;
      doWrite = true;
    }
    // actually write the buffer
    if (doWrite) {
      void *data = op->op_type == FSTORE
                       ? ((FStore *)op->additional_data)->data
                       : ((FResultData *)op->additional_data)->data;
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
    flogging(F_ERROR, msg);
  }
  // size for result
  result->dimensions = node_op->dimensions;
  result->shape = safe_mal<size_t>(result->dimensions);
  memcpy((void *)result->shape, (void *)node_op->shape,
         result->dimensions * sizeof(size_t));
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
                        "ms, execution took " + to_string(elapsed.count()) +
                        "ms");
  return newsucc;
}
