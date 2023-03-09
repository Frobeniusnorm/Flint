#include "../flint.h"
#include "../flint.hpp"
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "grad_test_cases.hpp"
int main(int argc, char **argv) {
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  flintInit(1, 0);
  int res = context.run();
  flintCleanup();
  return res;
}
