#include "../flint.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include <list>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <tuple>
#include <typeinfo>
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
void FlintBackend::init() {
  log(VERBOSE, "Initializing Flint");
  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
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
    char op = '\0';
    string type = typeString(node->operation->data_type);
    switch (node->operation->op_type) {
    case RESULTDATA: // already executed data that can be used as a parameter
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
      op = '+';
    case SUB:
      if (op != '\0')
        op = '-';
    case DIV:
      if (op != '\0')
        op = '/';
    case MUL:
      if (op != '\0')
        op = '*';
      // TODO check if one dimension > other dimension and add a for
      code = type + " " + name + " = v" + to_string(variable_index + 1) + " " +
             op + " v" + to_string(variable_index + 2) + ";\n" + code;
      break;
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
    log(WARNING, "Executing node with successor. Successor will be removed "
                 "from the Graph, make sure to save it.");
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
  string code = generateCode(node, parameters);

  return result;
}
