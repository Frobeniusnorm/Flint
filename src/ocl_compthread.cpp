#include "ocl_compthread.hpp"
#include "utils.hpp"
#include <algorithm>
#include <variant>

std::list<cl_program> OCLCompilerThread::eager_programs;
std::unordered_map<long, cl_kernel> OCLCompilerThread::eager_cache;
std::unordered_map<std::string, std::pair<cl_program, cl_kernel>>
    OCLCompilerThread::kernel_cache;
struct CompilePackage {
  FGraphNode *node;
  std::variant<int, std::string> data;
  std::thread *poisson_for = nullptr;
  bool operator==(const CompilePackage &other) {
    return node == other.node && data == other.data;
  }
};
static blocking_queue<CompilePackage> queue;
static std::list<CompilePackage> currently_compiling;
static std::mutex currently_compiling_mutex;
void OCLCompilerThread::enqueue_eager(FGraphNode *node, int hash) {
  queue.push_front({.node = node, .data = {hash}});
}
void OCLCompilerThread::enqueue_lazy(FGraphNode *node, std::string code) {
  queue.push_front({.node = node, .data = {code}});
}
bool OCLCompilerThread::is_enqueued_eager(FGraphNode *node, int hash) {
  std::unique_lock<std::mutex> lock(currently_compiling_mutex);
  const auto pack = CompilePackage{.node = node, .data = {hash}};
  return std::find(currently_compiling.begin(), currently_compiling.end(),
                   pack) == currently_compiling.end();
}
bool OCLCompilerThread::is_enqueued_lazy(FGraphNode *node, std::string code) {
  std::unique_lock<std::mutex> lock(currently_compiling_mutex);
  const auto pack = CompilePackage{.node = node, .data = {code}};
  return std::find(currently_compiling.begin(), currently_compiling.end(),
                   pack) == currently_compiling.end();
}
void OCLCompilerThread::compiler_thread(std::thread* thread) {
  while(true) {
    CompilePackage pack;
    {
      std::unique_lock<std::mutex> lock(currently_compiling_mutex);
      pack = queue.pop_front();
      if (pack.poisson_for && pack.poisson_for == thread) break;
    }
    if (pack.data.index() == 0)
      eager_compile(pack.node, std::get<0>(pack.data));
    else
      lazy_compile(pack.node, std::get<1>(pack.data));
  }
}
OCLCompilerThread::OCLCompilerThread() : thread(compiler_thread, &thread) {}
OCLCompilerThread::~OCLCompilerThread() {
  queue.push_front({.poisson_for = &thread});
  thread.join();
}
