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
#include "../operations/implementation.hpp"
#include "../utils.hpp"
#include "codegen.hpp"
#include <list>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

std::string generateEagerCode(FOperationType operation, FType res_type,
							  std::vector<FType> parameter_types,
							  std::string &kernel_name) {
	using namespace std;
	std::string type_info = to_string(res_type);
	for (FType t : parameter_types)
		type_info += to_string(t);
	kernel_name = string(fop_to_string[operation]) + type_info;
	Twine code =
		"#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n__kernel void " +
		kernel_name + "(__global " + type_string(res_type) +
		"* R, long num_entriesR";
	// generate parameters
	code += OperationImplementation::implementations[operation]
				->generate_ocl_parameters_eager(res_type, parameter_types);
	if (parameter_types.size() == 2)
		for (int i = 0; i < parameter_types.size(); i++)
			code += ", long inv_broad" + to_string(i);
	code += "){\nlong index = get_global_id(0);\n";
	// generate code
	code +=
		OperationImplementation::implementations[operation]->generate_ocl_eager(
			res_type, parameter_types);
	code += "\n}\n";
	return code;
}
