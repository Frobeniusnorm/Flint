#include "../flint.h"
#include "../flint.hpp"
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

template <typename T> static std::string printNode(FGraphNode *node) {
  std::string s = "";
  if (!node->result_data) {
    fExecuteGraph(node);
  }
  for (int i = 0; i < node->result_data->num_entries; i++)
    s += std::to_string(((T *)node->result_data->data)[i]) +
         (i == node->result_data->num_entries - 1 ? std::string("")
                                                  : std::string(", "));
  return s;
}
TEST_SUITE("Autodiff") {
  TEST_CASE("Two Times Matmul") {
    Flint::setLoggingLevel(2);
    enable_eager_execution();
    Tensor<double, 3> x = {{{1.0, 1.0}, {2.0, 3.0}}, {{4.0, 5.0}, {6.0, 7.0}}};
    Tensor<double, 2> y = {{3.0, -7.0}, {-1.0, 5.0}};
    Tensor<double, 3> z = {{{1, 1}, {2, 2}}, {{3, 3}, {-1, -1}}};
    Tensor<double, 3> w = (x.matmul(y)).matmul(z);
    disable_eager_execution();
    Tensor<double, 3> dx = w.gradient(x);
    CHECK_EQ(dx[0][0][0], -22);
    CHECK_EQ(dx[0][0][1], 18);
    CHECK_EQ(dx[0][1][0], -22);
    CHECK_EQ(dx[0][1][1], 18);
    CHECK_EQ(dx[1][0][0], 32);
    CHECK_EQ(dx[1][0][1], -16);
    CHECK_EQ(dx[1][1][0], 32);
    CHECK_EQ(dx[1][1][1], -16);
    Tensor<double, 2> dy = w.gradient(y);
    CHECK_EQ(dy[0][0], 66);
    CHECK_EQ(dy[0][1], -8);
    CHECK_EQ(dy[1][0], 80);
    CHECK_EQ(dy[1][1], -8);
    Tensor<double, 3> dz = w.gradient(z);
    CHECK_EQ(dz[0][0][0], 5);
    CHECK_EQ(dz[0][0][1], 5);
    CHECK_EQ(dz[0][1][0], -1);
    CHECK_EQ(dz[0][1][1], -1);
    CHECK_EQ(dz[1][0][0], 18);
    CHECK_EQ(dz[1][0][1], 18);
    CHECK_EQ(dz[1][1][0], -10);
    CHECK_EQ(dz[1][1][1], -10);
  }
}
int main(int argc, char **argv) {
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  flintInit(1, 0);
  int res = context.run();
  flintCleanup();
  return res;
}
