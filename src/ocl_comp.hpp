#include "../flint.h"
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
  // TODO hard drive caching of eager kernels here
  // TODO if we want to revisit a compiler thread ->
  //      - ONLY the compiler thread is allowed to compile code. This solves
  //      synchronizing problems with cache.
  //      - To do this we can insert in the queue at the back for "compile for
  //      future" kernels and in the front for "compile for now" kernels
#define MAX_NUMBER_PARAMS 2
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
