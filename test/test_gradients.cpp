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
  /*
   * TODO
   * i already checked that before an reduce every instruction must exist,
   * The problem is when two reduce operations are in parallel in an instruction
   * that does not require it in such a case i can be sure that the result
   * instruction is as large as the largest reduce_instruction, but the indices
   * for the other one are not correct
   *
   * How to solve this problem:
   * 1) batch reduces (or matmuls for that matter) only when their
   * dimensionality is equal
   *
   * 2) (better) include a modulo for dimensionality fix
   * in reduce operations
   */
  return res;
}
