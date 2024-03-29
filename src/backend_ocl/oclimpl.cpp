/* Copyright 2023 David Schwarzbeck
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "../../flint.h"
#include "../errors.hpp"
#include "../operations/implementation.hpp"
#include "../utils.hpp"
#include "codegen.hpp"
#include "comp.hpp"
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
using namespace std;
static const char *clCompilerOpts = "-cl-no-signed-zeros";
static void openclCallback(const char *errinfo, const void *privateinfo,
						   size_t cb, void *user_data) {
	flogging(F_WARNING, "{OpenCL} " + string(errinfo));
}

static bool initialized = false;
// opencl vars
static cl_context context;
static cl_command_queue clqueue;
static cl_device_id device;

FErrorType flintInit_gpu() {
	cl_platform_id platforms[10];
	cl_uint num_dev, num_plat;
	if (clGetPlatformIDs(10, &platforms[0], &num_plat) != CL_SUCCESS) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "clGetPlatformIds");
		return OCL_ERROR;
	}
	if (num_plat == 0) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR,
				 "Could not find any OpenCL Platform available! Please make "
				 "sure, you have setup your OpenCL driver right!");
		return OCL_ERROR;
	}
	flogging(F_VERBOSE, "Found " + to_string(num_plat) + " platforms!");
	device = NULL;
	// find suitable device
	char dev_name[128];
	size_t dev_name_size;
	char dev_vers[128];
	size_t dev_vers_size;
	char dev_vend[128];
	size_t dev_vend_size;
	cl_device_type dev_type;
	size_t dev_type_size;
	string dev_type_string;
	cl_uint dev_no_units;
	cl_uint highest_no_units = 0;
	for (int i = 0; i < num_plat; i++) {
		cl_device_id curr_dev;
		if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_DEFAULT, 1, &curr_dev,
						   &num_dev) != CL_SUCCESS) {
			flogging(F_WARNING, "clGetDeviceIDS did not return CL_SUCCESS!");
			clReleaseDevice(curr_dev);
			continue;
		}
		if (num_dev == 0) {
			flogging(F_WARNING, "Platform has no devices!");
			clReleaseDevice(curr_dev);
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
		clGetDeviceInfo(curr_dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
						&dev_no_units, nullptr);
		if (dev_no_units > highest_no_units) {
			highest_no_units = dev_no_units;
			if (device)
				clReleaseDevice(device);
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
		} else
			clReleaseDevice(curr_dev);
	}
	if (!device) {
		setErrorType(OCL_ERROR);
		flogging(
			F_ERROR,
			"Could not find any OpenCL devices available! Please make sure, "
			"you have setup your OpenCL driver right!");
		return OCL_ERROR;
	}
	string info = "Using " + dev_type_string + " '" +
				  string(dev_vend, dev_vend_size - 1) + "', '" +
				  string(dev_name, dev_name_size - 1) +
				  "' with OpenCL version " + string(dev_vers, dev_vers_size);
	flogging(F_INFO, info);
	int status = 0;
	context = clCreateContext(NULL, 1, &device, openclCallback, NULL, &status);
	if (status != CL_SUCCESS) {
		string err = "Could not create OpenCL context: ";
		setErrorType(OCL_ERROR);
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
			setErrorType(OUT_OF_MEMORY);
			err += "out of resources";
			break;
		case CL_OUT_OF_HOST_MEMORY:
			setErrorType(OUT_OF_MEMORY);
			err += "out of host memory";
			break;
		}
		flogging(F_ERROR, err);
		return OCL_ERROR;
	}
	clqueue =
		clCreateCommandQueueWithProperties(context, device, NULL, &status);
	if (status != CL_SUCCESS) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "clCreateCommandQueue " + to_string(status));
		return OCL_ERROR;
	}
	initialized = true;
	flogging(F_VERBOSE, "Flint GPU backend was initialized!");
	return NO_ERROR;
}
static cl_mem create_gpu_memory(FGraphNode *node, cl_mem_flags memory_type,
								size_t *total_size = nullptr) {
	cl_int err_code;
	size_t type_size_node = type_size(node->operation.data_type);
	size_t total_size_node = 1;
	if (node->operation.op_type != FGEN_CONSTANT)
		for (int i = 0; i < node->operation.dimensions; i++)
			total_size_node *= node->operation.shape[i];
	const cl_mem result_mem =
		clCreateBuffer(context, memory_type, total_size_node * type_size_node,
					   nullptr, &err_code);
	if (err_code == CL_OUT_OF_HOST_MEMORY) {
		setErrorType(OUT_OF_MEMORY);
		flogging(F_ERROR, "Not enough memory to create buffer!");
		return nullptr;
	}
	if (err_code != CL_SUCCESS) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Unknown Error while creating gpu memory!");
		return nullptr;
	}
	if (total_size)
		*total_size = total_size_node;
	return result_mem;
}
cl_mem OCLCompilerThread::copy_memory(const cl_mem other, size_t num_bytes,
									  cl_mem_flags memory_flags) {
	cl_int err_code;
	cl_mem mem =
		clCreateBuffer(context, memory_flags, num_bytes, nullptr, &err_code);
	if (err_code == CL_OUT_OF_HOST_MEMORY) {
		setErrorType(OUT_OF_MEMORY);
		flogging(F_ERROR, "Not enough memory to create buffer!");
		return nullptr;
	}
	if (err_code != CL_SUCCESS) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Unknown Error while creating gpu memory! " +
							  to_string(err_code));
		return nullptr;
	}
	clEnqueueCopyBuffer(clqueue, other, mem, 0, 0, num_bytes, 0, nullptr,
						nullptr);
	return mem;
}
#include <chrono>
#include <unordered_map>
cl_kernel OCLCompilerThread::eager_compile(FGraphNode *node, int hash) {
	cl_int err_code;
	cl_kernel kernel = nullptr;
	auto start = chrono::high_resolution_clock::now();
	// generate code for this operation for all datatypes (one kernel for each
	// parameter / return type combination) will be used to store all kernels
	// for this operation
	Twine code;
	// the kernel for `node`
	string our_kernel;
	// all generated kernels (for caching)
	vector<pair<int, string>> all_kernels;
	// match operation and generate code for all combinations
	const vector<vector<FType>> type_comb =
		OperationImplementation::implementations[node->operation.op_type]
			->kernel_type_combinations(node);
	for (vector<FType> types : type_comb) {
		const FType ret = types[0];
		types.erase(types.begin());
		string kernel_name;
		code +=
			generateEagerCode(node->operation.op_type, ret, types, kernel_name);
		bool correct = ret == node->operation.data_type;
		for (int i = 0; i < types.size(); i++)
			correct &= types[i] == node->predecessors[i]->operation.data_type;
		if (correct)
			our_kernel = kernel_name;
		all_kernels.push_back({OCLCompilerThread::generate_kernel_hash(
								   node->operation.op_type, ret, types),
							   kernel_name});
	}
	const string final_code = (string)code;
	flogging(F_DEBUG, string("Eager Kernel Generation for ") +
						  fop_to_string[node->operation.op_type] + ": " +
						  final_code);
	// generate kernel
	const char *code_data = final_code.data();
	const size_t code_length = final_code.length();
	cl_program prog = clCreateProgramWithSource(context, 1, &code_data,
												&code_length, &err_code);
	if (err_code == CL_OUT_OF_RESOURCES) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Out of resources while creating program!");
		return nullptr;
	}
	if (err_code == CL_OUT_OF_HOST_MEMORY) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Not enough memory to create program!");
		return nullptr;
	}
	// build program
	err_code =
		clBuildProgram(prog, 1, &device, clCompilerOpts, nullptr, nullptr);
	if (err_code == CL_INVALID_PROGRAM) {
		setErrorType(OCL_ERROR);
		flogging(
			F_ERROR,
			"Invalid Program was generated! Generated code: \"\n" + final_code +
				"\"\nPlease contact a developer and/or file a bug report.");
		return nullptr;
	} else if (err_code == CL_COMPILER_NOT_AVAILABLE) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Compiler of your GPU driver is not available!");
		return nullptr;
	} else if (err_code == CL_OUT_OF_HOST_MEMORY) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Not enough memory to build program!");
		return nullptr;
	} else if (err_code != CL_SUCCESS) {
		char build_log[4096];
		size_t actual_size = 0;
		clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 4096,
							  (void *)&build_log[0], &actual_size);
		setErrorType(OCL_ERROR);
		flogging(
			F_ERROR,
			"Unknown Error during program compilation! Generated code: \"\n" +
				final_code + "\nBuild Log:\n" + string(&build_log[0]) +
				"\"\nPlease contact a developer and/or file a bug report.");
		return nullptr;
	}
	// get kernel
	for (const auto &kernel_name : all_kernels) {
		cl_kernel curr =
			clCreateKernel(prog, kernel_name.second.c_str(), &err_code);
		if (err_code != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "kernel compilation failed! Kernel name: " +
								  kernel_name.second +
								  ", error_code: " + to_string(err_code));
			return nullptr;
		}
		OCLCompilerThread::eager_cache.insert({kernel_name.first, curr});
		if (kernel_name.first == hash) {
			kernel = curr;
		}
	}
	if (!kernel) {
		setErrorType(OCL_ERROR);
		for (int i = 0; i < node->num_predecessor; i++)
			std::cout << type_string(node->predecessors[i]->operation.data_type)
					  << std::endl;
		flogging(F_ERROR,
				 "something went horrible wrong for operation: " +
					 string(fop_to_string[node->operation.op_type]) +
					 " result type: " + type_string(node->operation.data_type));
		return nullptr;
	}
	OCLCompilerThread::eager_programs.push_back(prog);
	const chrono::duration<double, milli> elapsed =
		chrono::high_resolution_clock::now() - start;
	flogging(F_DEBUG, "Compilation took " + to_string(elapsed.count()) + "ms");
	return kernel;
}
// pushes additional per-parameter parameters to a opencl function
FGraphNode *fExecuteGraph_gpu_eagerly(FGraphNode *node) {
	if (node->result_data)
		return node;
	if (node->operation.op_type == FSTORE) {
		node->result_data = new FResultData();
		FStore *store = (FStore *)node->operation.additional_data;
		node->result_data->num_entries = store->num_entries;
		node->result_data->mem_id = store->mem_id;
		node->result_data->data = store->data;
		return node;
	}
	if ((node->operation.op_type == FLATTEN ||
		 node->operation.op_type == FRESHAPE) &&
		node->predecessors[0]->operation.op_type != FGEN_CONSTANT) {
		// just copy previous data
		const FGraphNode *prev = node->predecessors[0];
		void const *data;
		cl_mem gpu_data;
		size_t num_elems;
		if (prev->result_data) {
			data = prev->result_data->data;
			gpu_data = prev->result_data->mem_id;
			num_elems = prev->result_data->num_entries;
		} else if (prev->operation.op_type == FSTORE) {
			const FStore *store = (FStore *)prev->operation.additional_data;
			data = store->data;
			gpu_data = store->mem_id;
			num_elems = store->num_entries;
		}
		FResultData *rd = new FResultData();
		rd->data = nullptr;
		rd->num_entries = num_elems;
		rd->mem_id = nullptr;
		int type_s = type_size(node->operation.data_type);
		if (gpu_data) {
			rd->mem_id = OCLCompilerThread::copy_memory(
				gpu_data, type_s * num_elems, CL_MEM_READ_WRITE);
		} else if (data) {
			rd->data = malloc(type_s * num_elems);
			if (!rd->data) {
				setErrorType(OUT_OF_MEMORY);
				flogging(F_ERROR, "Not enough memory to store result! " +
									  to_string(num_elems));
				return nullptr;
			}
			memcpy(rd->data, data, type_s * num_elems);
		}
		node->result_data = rd;
		return node;
	}
	size_t inv_broad[2];
	if (node->num_predecessor == 2)
		calculate_divisor_for_inverse_broadcasting(
			node->predecessors[0], inv_broad[0], node->predecessors[1],
			inv_broad[1]);
	vector<FType> params_types(node->num_predecessor);
	for (int i = 0; i < node->num_predecessor; i++)
		params_types[i] = node->predecessors[i]->operation.data_type;
	// because the operation type should be at the same position
	int hash = OCLCompilerThread::generate_kernel_hash(
		node->operation.op_type, node->operation.data_type, params_types);
	const auto prog = OCLCompilerThread::eager_cache.find(hash);
	cl_kernel kernel = nullptr;
	cl_int err_code;
	list<cl_mem> to_free;
	// check if the kernel already exists or if it has to be generated
	if (prog == OCLCompilerThread::eager_cache.end()) {
		kernel = OCLCompilerThread::eager_compile(node, hash);
	} else {
		kernel = prog->second;
		flogging(F_DEBUG, "Loaded existing eager kernel");
	}
	// result data (may be recycle from predecessors)
	size_t total_size_node;
	cl_mem res_mem = nullptr;
	// load parameters
	const vector<bool> reusage =
		OperationImplementation::implementations[node->operation.op_type]
			->reuse_parameter_result(node);
	// collect memory objects from predecessors
	vector<cl_mem> mem_objs(node->num_predecessor);
	vector<size_t> mem_sizes(node->num_predecessor);
	vector<cl_event> write_events;
	for (int i = 0; i < node->num_predecessor; i++) {
		FGraphNode *pred = node->predecessors[i];
		const FOperation op = pred->operation;
		cl_mem mem_obj = nullptr;
		bool do_write = false;
		const size_t type_s = type_size(op.data_type);
		size_t total_size;
		cl_mem mem_id = nullptr;
		const bool recycle =
			!res_mem && pred->reference_counter == 1 && !reusage.empty() &&
			reusage[i] && (op.op_type != FSTORE || !node->gradient_data) &&
			((pred->result_data && pred->result_data->mem_id) ||
			 pred->operation.op_type == FSTORE) &&
			op.op_type != FGEN_CONSTANT;
		if (pred->result_data) {
			total_size = pred->operation.op_type == FGEN_CONSTANT
							 ? 1
							 : pred->result_data->num_entries;
			mem_id = pred->result_data->mem_id;
			if (recycle) {
				pred->result_data->mem_id = nullptr;
				if (pred->result_data->data)
					free(pred->result_data->data);
				delete pred->result_data;
				pred->result_data = nullptr;
				if (op.op_type == FSTORE)
					((FStore *)op.additional_data)->mem_id = nullptr;
			}
		}
		if (op.op_type == FSTORE && !mem_id) {
			total_size = ((FStore *)op.additional_data)->num_entries;
			mem_id = ((FStore *)op.additional_data)->mem_id;
			if (recycle)
				((FStore *)op.additional_data)->mem_id = nullptr;
		}
		if (mem_id) {
			mem_obj = mem_id;
		} else {
			mem_obj = create_gpu_memory(pred, CL_MEM_READ_WRITE, &total_size);
			// only set actual gpu memory if it is not recycled
			if (!recycle) {
				if (op.op_type == FSTORE) {
					((FStore *)op.additional_data)->mem_id = mem_obj;
					if (pred->result_data)
						pred->result_data->mem_id = mem_obj;
				} else {
					pred->result_data->mem_id = mem_obj;
				}
			}
			do_write = true;
		}
		mem_sizes[i] = total_size;
		mem_objs[i] = mem_obj;
		if (recycle) {
			total_size_node = total_size;
			res_mem = mem_obj;
		}
		if (do_write) {
			void *data =
				op.op_type == FSTORE ? ((FStore *)op.additional_data)->data
				: op.op_type == FGEN_CONSTANT ? op.additional_data
											  : pred->result_data->data;
			if (!data) {
				flogging(F_WARNING,
						 "No gpu memory is found, but no cpu either! " +
							 to_string((long)pred->result_data->data) + ", " +
							 to_string((long)pred->result_data->mem_id) + ", " +
							 fop_to_string[op.op_type]);
			}
			cl_event write_event;
			err_code = clEnqueueWriteBuffer(clqueue, mem_obj, CL_FALSE, 0,
											total_size * type_s, data, 0,
											nullptr, &write_event);
			if (err_code != CL_SUCCESS) {
				string msg = "Unknown Error while loading data to GPU! Error: ";
				setErrorType(OCL_ERROR);
				if (err_code == CL_OUT_OF_HOST_MEMORY) {
					msg = "Not enough memory to load data to GPU! ";
					setErrorType(OUT_OF_MEMORY);
				}
				flogging(F_ERROR, msg + to_string(err_code));
				return nullptr;
			}
			write_events.push_back(write_event);
		}
	}
	// result buffer
	if (!res_mem) {
		res_mem = create_gpu_memory(node, CL_MEM_READ_WRITE, &total_size_node);
		// zero result
		cl_event zero_event;
		const long zero_pattern = 0;
		const long type_size_node = type_size(node->operation.data_type);
		clEnqueueFillBuffer(clqueue, res_mem, &zero_pattern, type_size_node, 0,
							total_size_node * type_size_node, 0, nullptr,
							&zero_event);
		write_events.push_back(zero_event);
	}
	node->result_data = new FResultData();
	node->result_data->mem_id = res_mem;
	node->result_data->num_entries = total_size_node;
	node->result_data->data = nullptr;
	int par_index = 0;
	if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&res_mem) !=
		CL_SUCCESS) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Could not load Argument to kernel!");
		return nullptr;
	}
	if (clSetKernelArg(kernel, par_index++, sizeof(long),
					   (void *)&total_size_node) != CL_SUCCESS) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Could not load Argument to kernel!");
		return nullptr;
	}
	// process parameters i.e. predecessors
	for (int i = 0; i < node->num_predecessor; i++) {
		FGraphNode *pred = node->predecessors[i];
		const FOperation op = pred->operation;
		cl_mem mem_obj = mem_objs[i];
		if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
						   (void *)&mem_obj) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return nullptr;
		}
		// push total element size
		if (clSetKernelArg(kernel, par_index++, sizeof(long),
						   (void *)&mem_sizes[i]) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return nullptr;
		}
		// push per-parameter values
		OperationImplementation::implementations[node->operation.op_type]
			->push_parameter_kernel_parameters(node, pred, kernel, context,
											   par_index, to_free);
	}
	// parameters for functions that dont set them per parent
	OperationImplementation::implementations[node->operation.op_type]
		->push_additional_kernel_parameters(node, kernel, context, par_index,
											to_free);
	// broadcasting values
	if (node->num_predecessor == 2) {
		for (int i = 0; i < node->num_predecessor; i++) {
			if (clSetKernelArg(kernel, par_index++, sizeof(long),
							   (void *)&inv_broad[i]) != CL_SUCCESS) {
				setErrorType(OCL_ERROR);
				flogging(F_ERROR, "Could not load Argument to kernel!");
				return nullptr;
			}
		}
	}
	// execute it
	size_t execution_size =
		OperationImplementation::implementations[node->operation.op_type]
			->deploy_as_many_elements(node);
	// usually divisable by two for this
	err_code = clEnqueueNDRangeKernel(
		clqueue, kernel, 1, nullptr, &execution_size, nullptr,
		write_events.size(), write_events.data(), nullptr);
	for (cl_event ev : write_events)
		clReleaseEvent(ev);
	if (err_code != CL_SUCCESS) {
		setErrorType(OCL_ERROR);
		string msg;
		switch (err_code) {
		case CL_OUT_OF_HOST_MEMORY:
			setErrorType(OUT_OF_MEMORY);
			msg = "Not enough memory to execute kernel!";
			break;
		case CL_OUT_OF_RESOURCES:
			setErrorType(OUT_OF_MEMORY);
			msg = "Out of resources!";
			break;
		default:
			msg = "Unknown Error during kernel execution! code: " +
				  to_string(err_code);
			break;
		}
		flogging(F_ERROR, msg);
		return nullptr;
	}
	for (cl_mem tfn : to_free)
		clReleaseMemObject(tfn);
	return node;
}
cl_kernel OCLCompilerThread::lazy_compile(FGraphNode *node, string code) {
	using namespace std;
	cl_kernel kernel;
	cl_int err_code;
	// create program
	const char *code_data = code.data();
	const size_t code_length = code.length();
	cl_program prog = clCreateProgramWithSource(context, 1, &code_data,
												&code_length, &err_code);
	if (err_code == CL_OUT_OF_RESOURCES) {
		setErrorType(OUT_OF_MEMORY);
		flogging(F_ERROR, "Out of resources while creating program!");
		return nullptr;
	}
	if (err_code == CL_OUT_OF_HOST_MEMORY) {
		setErrorType(OUT_OF_MEMORY);
		flogging(F_ERROR, "Not enough memory to create program!");
		return nullptr;
	}
	// build program
	err_code =
		clBuildProgram(prog, 1, &device, clCompilerOpts, nullptr, nullptr);
	if (err_code == CL_INVALID_PROGRAM) {
		setErrorType(OCL_ERROR);
		flogging(
			F_ERROR,
			"Invalid Program was generated! Generated code: \"\n" + code +
				"\"\nPlease contact a developer and/or file a bug report.");
		return nullptr;
	} else if (err_code == CL_COMPILER_NOT_AVAILABLE) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Compiler of your GPU driver is not available!");
		return nullptr;
	} else if (err_code == CL_OUT_OF_HOST_MEMORY) {
		setErrorType(OUT_OF_MEMORY);
		flogging(F_ERROR, "Not enough memory to build program!");
		return nullptr;
	} else if (err_code != CL_SUCCESS) {
		char build_log[4096];
		size_t actual_size = 0;
		clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 4096,
							  (void *)&build_log[0], &actual_size);
		setErrorType(OCL_ERROR);
		flogging(
			F_ERROR,
			"Unknown Error during program compilation! Generated code: \"\n" +
				code + "\nBuild Log:\n" + string(&build_log[0]) +
				"\"\nPlease contact a developer and/or file a bug report.");
		return nullptr;
	}
	// get kernel
	kernel = clCreateKernel(prog, "execute_graph", &err_code);
	if (err_code != CL_SUCCESS) {
		// try to clear cache and retry
		for (auto &k : OCLCompilerThread::kernel_cache) {
			clReleaseKernel(k.second.second);
			clReleaseProgram(k.second.first);
		}
		kernel = clCreateKernel(prog, "execute_graph", &err_code);
		if (err_code != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "kernel compilation failed (lazy)! " +
								  to_string(err_code));
			return nullptr;
		}
	}
	OCLCompilerThread::kernel_cache.insert({code, {prog, kernel}});
	return kernel;
}
void OCLCompilerThread::memory_barrier() { clFinish(clqueue); }
FResultData *fSyncMemory(FGraphNode *node) {
	void **store_data = nullptr;
	if (node->result_data && node->result_data->data)
		return node->result_data;
	if (node->operation.op_type == FSTORE) {
		FStore *store = (FStore *)node->operation.additional_data;
		if (!node->result_data) {
			node->result_data = new FResultData();
			node->result_data->num_entries = store->num_entries;
		}
		if (!node->result_data->mem_id)
			node->result_data->mem_id = store->mem_id;
		if (!node->result_data->data)
			node->result_data->data = store->data;
		store_data = &store->data;
	}
	FResultData *res = node->result_data;
	if (res && res->mem_id && !res->data) {
		// read result to cpu
		int type_size_node = type_size(node->operation.data_type);
		res->data = malloc(res->num_entries * type_size_node);
		if (store_data)
			*store_data = res->data;
		res->num_entries = res->num_entries;
		if (!node->result_data->data) {
			setErrorType(OUT_OF_MEMORY);
			flogging(F_ERROR, "Not enough memory to store result!");
			return nullptr;
		}
		// wait for result
		cl_int err_code = clEnqueueReadBuffer(clqueue, res->mem_id, CL_TRUE, 0,
											  res->num_entries * type_size_node,
											  res->data, 0, nullptr, nullptr);
		if (err_code != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			string msg =
				"Unknown Error while reading the result! Error Code: " +
				to_string(err_code);
			if (err_code == CL_OUT_OF_HOST_MEMORY) {
				setErrorType(OUT_OF_MEMORY);
				msg = "Not enough memory to read result!";
			}
			flogging(F_ERROR, msg);
			return nullptr;
		}
	}
	return res;
}
/**
 * Calculates per kernel parameter if its result is reusable
 */
static vector<bool>
find_reusable_parameters(const FGraphNode *node,
						 const list<pair<FGraphNode *, string>> params) {
	vector<bool> result(params.size(), false);
	list<const FGraphNode *> todo;
	todo.push_front(node);
	while (!todo.empty()) {
		const FGraphNode *curr = todo.front();
		todo.pop_front();
		const vector<bool> reusage =
			OperationImplementation::implementations[curr->operation.op_type]
				->reuse_parameter_result(curr);
		for (int i = 0; i < curr->num_predecessor; i++) {
			if (!reusage.empty() && reusage[i]) {
				const FGraphNode *pred = curr->predecessors[i];
				bool allow_recycle = true;
				if (pred->operation.op_type == FSTORE) {
					allow_recycle = curr->gradient_data == nullptr;
				}
				if (allow_recycle) {
					int j = 0;
					for (const auto &[param, name] : params) {
						if (pred == param)
							result[j] = true;
						j++;
					}
					todo.push_back(pred);
				}
			}
		}
	}
	return result;
}
FGraphNode *fExecuteGraph_gpu(FGraphNode *node) {
	if (!initialized) {
		flintInit_gpu();
	}
	{
		if (node->operation.op_type == FSTORE) {
			node->result_data = new FResultData();
			FStore *store = (FStore *)node->operation.additional_data;
			node->result_data->num_entries = store->num_entries;
			node->result_data->mem_id = store->mem_id;
			node->result_data->data = store->data;
		}
		if (node->result_data)
			return node;
	}
	// eager if all parameters have result
	bool all_have_result = true; // TODO should be true, but does not work,
								 // probably because of constants
	for (int i = 0; i < node->num_predecessor; i++) {
		const FGraphNode *pred = node->predecessors[i];
		if (!pred->result_data ||
			(!pred->result_data->data && !pred->result_data->mem_id)) {
			all_have_result = false;
			break;
		}
	}
	// then execute eagerly
	if (all_have_result)
		return fExecuteGraph_gpu_eagerly(node);
	auto start = chrono::high_resolution_clock::now();
	FResultData *resultData = new FResultData();
	const FOperation node_op = node->operation;
	size_t total_size_node = 1;
	if (node_op.op_type != FGEN_CONSTANT)
		for (int i = 0; i < node_op.dimensions; i++)
			total_size_node *= node_op.shape[i];
	// calculate Code and Parameters
	list<pair<FGraphNode *, string>> parameters;
	string graph_code = generateCode(node, parameters);
	string code =
		"#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n__kernel void "
		"execute_graph(__global ";
	code += type_string(node->operation.data_type);
	code += " *R";
	// insert parameters
	for (auto &[op, name] : parameters)
		code += ", __global const " + type_string(op->operation.data_type) +
				" *" + name;
	code += "){\n";
	// add the execution code
	code += graph_code;
	// store result
	code += "R[index] = v0;\n}";
	// don't create code when in cache
	auto cache_val = OCLCompilerThread::kernel_cache.find(code);
	cl_kernel kernel = nullptr;
	cl_int err_code;
	chrono::duration<double, milli> elapsed =
		chrono::high_resolution_clock::now() - start;
	if (cache_val == OCLCompilerThread::kernel_cache.end()) {
		flogging(F_DEBUG, "code generation finished (in " +
							  to_string(elapsed.count()) + " ms): \n" + code);
		kernel = OCLCompilerThread::lazy_compile(node, code);
	} else {
		flogging(F_DEBUG, "code from cache");
		kernel = cache_val->second.second;
	}
	chrono::duration<double, milli> compilation_time =
		chrono::high_resolution_clock::now() - start;
	start = chrono::high_resolution_clock::now();
	// result buffer
	size_t type_size_node = type_size(node_op.data_type);
	cl_mem result_mem = nullptr;
	vector<cl_event> writeEvents;
	// upload or link parameters
	cl_mem mem_objs[parameters.size()];
	const vector<bool> reusable = find_reusable_parameters(node, parameters);
	{
		int index = 0;
		for (auto &[gn, name] : parameters) {
			const FOperation op = gn->operation;
			const bool recycle = !result_mem && gn->reference_counter == 1 &&
								 reusable[index] && op.op_type != FGEN_CONSTANT;
			// The problem here: optimized memory is a store
			cl_mem mem_obj = nullptr;
			bool do_write = false;
			const size_t type_s = type_size(op.data_type);
			const size_t total_size =
				op.op_type == FSTORE
					? ((FStore *)op.additional_data)->num_entries
					: (op.op_type == FGEN_CONSTANT
						   ? 1
						   : gn->result_data->num_entries);
			cl_mem mem_id = gn->result_data ? gn->result_data->mem_id : nullptr;
			if (!mem_id && op.op_type == FSTORE)
				mem_id = ((FStore *)op.additional_data)->mem_id;
			if (op.op_type == FSTORE && recycle && mem_id) {
				((FStore *)op.additional_data)->mem_id = nullptr;
			}
			if (mem_id) {
				mem_obj = mem_id;
				if (recycle) {
					gn->result_data->mem_id = nullptr;
					if (!gn->result_data->data) {
						delete gn->result_data;
						gn->result_data = nullptr;
					}
				}
			} else {
				mem_obj =
					clCreateBuffer(context, CL_MEM_READ_WRITE,
								   total_size * type_s, nullptr, &err_code);
				if (err_code == CL_OUT_OF_HOST_MEMORY) {
					setErrorType(OUT_OF_MEMORY);
					flogging(F_ERROR, "Not enough memory to create buffer!");
					return nullptr;
				}
				if (op.op_type == FSTORE && !recycle)
					((FStore *)op.additional_data)->mem_id = mem_obj;
				if (op.op_type == FGEN_CONSTANT && !gn->result_data &&
					!recycle) {
					gn->result_data = new FResultData();
					gn->result_data->data = nullptr;
					gn->result_data->num_entries = 1;
				}
				if (gn->result_data && !recycle)
					gn->result_data->mem_id = mem_obj;
				do_write = true;
			}
			mem_objs[index++] = mem_obj;
			if (recycle) {
				result_mem = mem_obj;
			}
			// actually write the buffer
			if (do_write) {
				void *data = op.op_type == FSTORE
								 ? ((FStore *)op.additional_data)->data
							 : gn->operation.op_type == FGEN_CONSTANT
								 ? gn->operation.additional_data
								 : gn->result_data->data;
				if (!data)
					flogging(F_ERROR, "parameter has no data!");
				writeEvents.emplace_back();
				err_code = clEnqueueWriteBuffer(
					clqueue, mem_obj, CL_FALSE, 0, total_size * type_s, data, 0,
					nullptr, &writeEvents[writeEvents.size() - 1]);
				if (err_code != CL_SUCCESS) {
					string msg = "Unknown Error while loading data to GPU!";
					flogging(F_ERROR, msg);
					setErrorType(OCL_ERROR);
					if (err_code == CL_OUT_OF_HOST_MEMORY) {
						msg = "Not enough memory to load data to GPU!";
					}
					return nullptr;
				}
			}
		}
	}
	// link resource memory
	if (!result_mem) {
		result_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
									total_size_node * type_size_node, nullptr,
									&err_code);
		if (err_code == CL_OUT_OF_HOST_MEMORY) {
			setErrorType(OUT_OF_MEMORY);
			flogging(F_ERROR, "Not enough memory to create buffer!");
			return nullptr;
		}
	}
	resultData->mem_id = result_mem;
	if ((err_code = clSetKernelArg(kernel, 0, sizeof(cl_mem),
								   (void *)&result_mem)) != CL_SUCCESS) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Could not set Kernel Argument for the result! " +
							  to_string(err_code));
		return nullptr;
	}
	// link parameter memory
	for (int i = 0; i < parameters.size(); i++) {
		if (clSetKernelArg(kernel, i + 1, sizeof(cl_mem),
						   (void *)&mem_objs[i]) != CL_SUCCESS) {
			setErrorType(OCL_ERROR);
			flogging(F_ERROR, "Could not load Argument to kernel!");
			return nullptr;
		}
	}
	// execute kernel
	const size_t global_size = total_size_node;

	err_code = clEnqueueNDRangeKernel(clqueue, kernel, 1, nullptr, &global_size,
									  nullptr, writeEvents.size(),
									  writeEvents.data(), nullptr);
	for (cl_event ev : writeEvents)
		clReleaseEvent(ev);
	if (err_code != CL_SUCCESS) {
		string msg;
		setErrorType(OCL_ERROR);
		switch (err_code) {
		case CL_OUT_OF_HOST_MEMORY:
			msg = "Not enough memory to execute kernel!";
			setErrorType(OUT_OF_MEMORY);
			break;
		case CL_OUT_OF_RESOURCES:
			msg = "Out of resources!";
			setErrorType(OUT_OF_MEMORY);
			break;
		default:
			msg =
				"Unknown Error during kernel execution! " + to_string(err_code);
			break;
		}
		flogging(F_ERROR, msg);
		return nullptr;
	}
	resultData->num_entries = total_size_node;
	OCLCompilerThread::memory_barrier();
	elapsed = chrono::high_resolution_clock::now() - start;
	flogging(F_DEBUG, "compilation took " +
						  to_string(compilation_time.count()) +
						  "ms, execution took " + to_string(elapsed.count()) +
						  " for " + to_string(global_size) + " elements");
	node->result_data = resultData;
	return node;
}
FErrorType flintCleanup_gpu() {
	if (initialized) {
		flogging(F_DEBUG, "Cleaning up GPU Backend");
		clReleaseDevice(device);
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
		clReleaseContext(context);
	}
	return NO_ERROR;
}
