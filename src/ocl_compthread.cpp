#include "ocl_compthread.hpp"
std::list<cl_program> OCLCompilerThread::eager_programs;
std::unordered_map<long, cl_kernel> OCLCompilerThread::eager_cache;
std::unordered_map<std::string, std::pair<cl_program, cl_kernel>> OCLCompilerThread::kernel_cache;

void OCLCompilerThread::enqueue_eager(FGraphNode *node, int hash){
  return; // TODO
}
void OCLCompilerThread::enqueue_lazy(FGraphNode *node, std::string code){
  return; // TODO
}
bool OCLCompilerThread::is_enqueued_eager(FGraphNode *node, int hash){
  return false; // TODO
}
bool OCLCompilerThread::is_enqueued_lazy(FGraphNode *node, int hash){
  return false; // TODO
}
