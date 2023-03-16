#include "../flint.h"
#include "../flint.hpp"
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "grad_test_cases.hpp"
int main(int argc, char **argv) {
  bool doCPU = false, doGPU = false;
  for (int i = 0; i < argc; i++) {
    std::string arg(argv[i]);
    if (arg == "cpu")
      doCPU = true;
    if (arg == "gpu")
      doGPU = true;
  }
  if (!doCPU && !doGPU) {
    doCPU = doGPU = true;
  }
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  int res;
  if (doCPU) {
    flintInit(1, 0);
    res = context.run();
    flintCleanup();
  }
  if (doGPU) {
    flintInit(0, 1);
    res = context.run();
    flintCleanup();
  }
  return res;
}
