#include "../flint.hpp"
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
#include <vector>
#define CL_TARGET_OPENCL_VERSION 200
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
using namespace FlintBackend;
static void openclCallback(const char *errinfo, const void *privateinfo,
                           size_t cb, void *user_data) {
  log(WARNING, "{OpenCL} " + std::string(errinfo));
}
static bool initialized = false;
// opencl vars
static cl_context context;
static cl_command_queue queue;
static cl_device_id device;
void FlintBackend::init() {
  log(VERBOSE, "Initializing Flint");
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
  log(VERBOSE, "Flint was initialized!");
}
void FlintBackend::cleanup() {
  if (initialized) {
    clReleaseDevice(device);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
  }
}

static std::string typeString(Type t) {
  switch (t) {
  case INT32:
    return "int";
  case INT64:
    return "long";
  case FLOAT32:
    return "float";
  case FLOAT64:
    return "double";
  }
  return "";
}
static size_t typeSize(Type t) {
  switch (t) {
  case INT32:
    return sizeof(int);
  case INT64:
    return sizeof(long);
  case FLOAT32:
    return sizeof(float);
  case FLOAT64:
    return sizeof(double);
  }
  return 1;
}
void FlintBackend::setLoggingLevel(int level) { setLoggerLevel(level); }

static std::string
generateCode(GraphNode *node,
             std::list<std::pair<Operation *, std::string>> &parameters) {
  using namespace std;
  list<pair<GraphNode *, string>> todo;
  int variable_index = 0;
  string code = "";
  todo.push_front({node, "v0"});
  while (!todo.empty()) {
    // take from queue
    const auto [node, name] = todo.front();
    todo.pop_front();
    // write code
    string type = typeString(node->operation->data_type);
    switch (node->operation->op_type) {
    case RESULTDATA:
    case STORE:
      parameters.push_front({node->operation, name});
      break;
    case CONST:
      switch (node->operation->data_type) {
      case INT32: {
        Const<int> *actcst = (Const<int> *)node->operation;
        code =
            type + " " + name + " = " + to_string(actcst->value) + ";\n" + code;
      } break;
      case INT64: {
        Const<long> *actcst = (Const<long> *)node->operation;
        code =
            type + " " + name + " = " + to_string(actcst->value) + ";\n" + code;
      } break;
      case FLOAT64: {
        Const<double> *actcst = (Const<double> *)node->operation;
        code =
            type + " " + name + " = " + to_string(actcst->value) + ";\n" + code;
      } break;
      case FLOAT32: {
        Const<float> *actcst = (Const<float> *)node->operation;
        code =
            type + " " + name + " = " + to_string(actcst->value) + ";\n" + code;
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
    }
    // push predecessors
    for (int i = 0; i < node->num_predecessor; i++)
      todo.push_front(
          {node->predecessors[i], "v" + to_string(++variable_index)});
  }
  return code;
}

ResultData *FlintBackend::executeGraph(GraphNode *node) {
  if (node->successor != NULL) {
    log(WARNING, "Executing node with successor. Successor tree will not be "
                 "executed and not linked to the result!");
  }
  ResultData *result = new ResultData();
  GraphNode *newsucc = new GraphNode();
  newsucc->successor = nullptr;
  newsucc->num_predecessor = 1;
  newsucc->predecessors = safe_mal<GraphNode *>(1);
  newsucc->predecessors[0] = node;
  newsucc->operation = result;
  node->successor = newsucc;
  // calculate Code and Parameters
  using namespace std;
  list<pair<Operation *, string>> parameters;
  string graph_code = generateCode(node, parameters);
  string code = "__kernel void execute_graph(__global ";
  code += typeString(node->operation->data_type);
  code += " *R";
  // insert parameters
  int par_idx = 0;
  for (auto &[op, name] : parameters)
    code += ", __global__ const " + typeString(op->data_type) + " *P" +
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
      // we divide through the sizes of the dimensions that are not shared by op
      int factor = 1;
      for (int i = op->dimensions; i < node->operation->dimensions; i++)
        factor *= node->operation->shape[i];
      indx_mod = "/" + to_string(factor);
    }
    code += "P" + to_string(par_idx) + "[index" + indx_mod + "];\n";
    par_idx++;
  }
  // add the execution code
  code += graph_code;
  // store result
  code += "R[index] = v0;\n}";
  // create program
  cl_int err_code;
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
    log(ERROR, "Invalid Program was generated! Generated code: \"\n" + code +
                   "\"\nPlease contact a developer and/or file a bug report.");
  else if (err_code == CL_COMPILER_NOT_AVAILABLE)
    log(ERROR, "Compiler of your GPU driver is not available!");
  else if (err_code == CL_OUT_OF_HOST_MEMORY)
    log(ERROR, "Not enough memory to create program!");
  // get kernel
  cl_kernel kernel = clCreateKernel(prog, "execute_graph", &err_code);
  if (err_code == CL_OUT_OF_HOST_MEMORY)
    log(ERROR, "Not enough memory to create program!");
  // TODO: memorizing kernels
  // result buffer
  Operation *node_op = node->operation;
  size_t total_size_node = 1;
  for (int i = 0; i < node_op->dimensions; i++)
    total_size_node *= node_op->shape[i];
  size_t type_size_node = typeSize(node_op->data_type);
  cl_mem result_mem =
      clCreateBuffer(context, CL_MEM_READ_WRITE,
                     total_size_node * type_size_node, nullptr, &err_code);
  node_op->mem_id = result_mem;
  if (err_code == CL_OUT_OF_HOST_MEMORY)
    log(ERROR, "Not enough memory to create buffer!");
  int index = 1;
  for (auto &[op, name] : parameters) {
    // TODO keep track of when data in Store is changed
    cl_mem mem_obj = nullptr;
    bool doWrite =
        op->op_type ==
        STORE; // can always be changed, ResultData only changes with gpu data
    size_t type_size = typeSize(op->data_type);
    size_t total_size = op->op_type == STORE ? ((Store *)op)->num_entries
                                             : ((ResultData *)op)->num_entries;
    if (op->mem_id &&
        op != node_op) { // second case is needed for the case that the result
                         // node is a parameter at the same time
      mem_obj = op->mem_id;
    } else {
      mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                               total_size * type_size, nullptr, &err_code);
      if (err_code == CL_OUT_OF_HOST_MEMORY)
        log(ERROR, "Not enough memory to create buffer!");
      op->mem_id = mem_obj;
      doWrite = true;
    }
    // actually write the buffer
    if (doWrite) {
      void *data =
          op->op_type == STORE ? ((Store *)op)->data : ((ResultData *)op)->data;
      err_code = clEnqueueWriteBuffer(queue, mem_obj, CL_TRUE, 0,
                                      total_size * type_size, data, 0, nullptr,
                                      nullptr);
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
  err_code = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size,
                                    nullptr, 0, nullptr, nullptr);
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
  result->data = malloc(total_size_node);
  if (!result->data)
    log(ERROR, "Not enough memory to store result!");
  // wait for result
  err_code = clEnqueueReadBuffer(queue, result_mem, CL_TRUE, 0, total_size_node,
                                 (void *)result->data, 0, nullptr, nullptr);
  if (err_code != CL_SUCCESS) {
    string msg = "Unknown Error while reading the result!";
    if (err_code == CL_OUT_OF_HOST_MEMORY)
      msg = "Not enough memory to read result!";
    log(ERROR, msg);
  }
  return result;
}
