#include "ocl_comp.hpp"
#include "utils.hpp"
#include <algorithm>
#include <variant>

std::list<cl_program> OCLCompilerThread::eager_programs;
std::unordered_map<long, cl_kernel> OCLCompilerThread::eager_cache;
std::unordered_map<std::string, std::pair<cl_program, cl_kernel>>
    OCLCompilerThread::kernel_cache;
