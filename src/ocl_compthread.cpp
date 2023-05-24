#include "ocl_compthread.hpp"
#include "src/utils.hpp"
#include <variant>
std::list<cl_program> OCLCompilerThread::eager_programs;
std::unordered_map<long, cl_kernel> OCLCompilerThread::eager_cache;
std::unordered_map<std::string, std::pair<cl_program, cl_kernel>>
    OCLCompilerThread::kernel_cache;
struct CompilePackage {
  FGraphNode *node;
  std::variant<int, std::string> data;
  std::thread *poisson_for = nullptr;
};
static blocking_queue<CompilePackage> queue;
void OCLCompilerThread::enqueue_eager(FGraphNode *node, int hash) {
  queue.push_front({.node = node, .data = {hash}});
}
void OCLCompilerThread::enqueue_lazy(FGraphNode *node, std::string code) {
  queue.push_front({.node = node, .data = {code}});
}
bool OCLCompilerThread::is_enqueued_eager(FGraphNode *node, int hash) {
  return false; // TODO
}
bool OCLCompilerThread::is_enqueued_lazy(FGraphNode *node, std::string code) {
  return false; // TODO
}
void OCLCompilerThread::compiler_thread() {
  // TODO
}
OCLCompilerThread::OCLCompilerThread() : thread(compiler_thread) {}
OCLCompilerThread::~OCLCompilerThread() {
  queue.push_front({.poisson_for = &thread});
  thread.join();
}
