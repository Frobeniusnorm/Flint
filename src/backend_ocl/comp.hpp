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

  This file includes the OpenCL compiler front end (and maybe someday
  asynchronous compiler and cache)
*/
#include "../../flint.h"
#include <list>
#include <thread>
#include <unordered_map>
#include <vector>

struct OCLCompilerThread {
  static std::list<cl_program> eager_programs;
  static std::unordered_map<long, cl_kernel> eager_cache;
  static std::unordered_map<std::string, std::pair<cl_program, cl_kernel>>
      kernel_cache;
  static cl_kernel eager_compile(FGraphNode *node, int hash);
  static cl_kernel lazy_compile(FGraphNode *node, std::string code);
  static cl_mem copy_memory(const cl_mem other, size_t num_bytes,
                            cl_mem_flags memory_flags);
  static void memory_barrier();
  // TODO hard drive caching of eager kernels here
  // TODO if we want to revisit a compiler thread ->
  //      - ONLY the compiler thread is allowed to compile code. This solves
  //      synchronizing problems with cache.
  //      - To do this we can insert in the queue at the back for "compile for
  //      future" kernels and in the front for "compile for now" kernels
#define MAX_NUMBER_PARAMS 3
  static int generateKernelHash(FOperationType operation, FType return_type,
                                std::vector<FType> params) {
    int hash = (operation << 3) |
               return_type; // 4 types, 2 bits are enough to decode them
    for (int i = 0; i < params.size(); i++)
      hash = (hash << 3) | params[i];
    for (int i = 0; i < MAX_NUMBER_PARAMS - params.size(); i++)
      hash = hash << 3;
    return hash;
  }
};
