#include "../flint.h"
#include "../flint.hpp"
#include "../flint_helper.hpp"
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "grad_test_cases.hpp"
int main(int argc, char **argv) {
  bool doCPU = false, doGPU = false, eager = false;
  for (int i = 0; i < argc; i++) {
    std::string arg(argv[i]);
    if (arg == "cpu")
      doCPU = true;
    if (arg == "gpu")
      doGPU = true;
    if (arg == "eager")
      eager = true;
  }
  if (!doCPU && !doGPU) {
    doCPU = doGPU = true;
  }
  fSetLoggingLevel(F_DEBUG);
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  int res;
  if (doCPU) {
    flintInit(FLINT_BACKEND_ONLY_CPU);
    if (eager)
      fEnableEagerExecution();
    res = context.run();
    flintCleanup();
  }
  if (doGPU) {
    flintInit(FLINT_BACKEND_ONLY_GPU);
    if (eager)
      fEnableEagerExecution();
    res = context.run();
    flintCleanup();
  }
  return res;
}
