#include "../flint.hpp"
#include "logger.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#define CL_TARGET_OPENCL_VERSION 200
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
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

void FlintBackend::setLoggingLevel(int level) { setLoggerLevel(level); }
