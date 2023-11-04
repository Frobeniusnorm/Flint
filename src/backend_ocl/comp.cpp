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

  This file includes the OpenCL compiler front end symbols
*/
#include "comp.hpp"
#include "../utils.hpp"
#include <algorithm>
#include <variant>

std::list<cl_program> OCLCompilerThread::eager_programs;
std::unordered_map<long, cl_kernel> OCLCompilerThread::eager_cache;
std::unordered_map<std::string, std::pair<cl_program, cl_kernel>>
	OCLCompilerThread::kernel_cache;
