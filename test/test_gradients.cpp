#include "../flint.h"
#include "../flint.hpp"
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

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
    Tensor<double, 3> zy = z.matmul(y);
    w = (x.matmul(y)).matmul(zy);
    dy = w.gradient(y);
    CHECK_EQ(dy[0][0], 67);
    CHECK_EQ(dy[0][1], 67);
    CHECK_EQ(dy[1][0], 67);
    CHECK_EQ(dy[1][1], 67);
    dx = w.gradient(x);
    CHECK_EQ(dx[0][0][0], 0);
    CHECK_EQ(dx[1][1][1], 0);
    dz = w.gradient(z);
    CHECK_EQ(dz[0][0][0], -20);
    CHECK_EQ(dz[0][0][1], 20);
    CHECK_EQ(dz[0][1][0], 4);
    CHECK_EQ(dz[0][1][1], -4);
    CHECK_EQ(dz[1][0][0], -72);
    CHECK_EQ(dz[1][0][1], 72);
    CHECK_EQ(dz[1][1][0], 40);
    CHECK_EQ(dz[1][1][1], -40);
  }
  TEST_CASE("Add, Mul, Matmul") {
    Tensor<double, 3> x = {{{1.0, 1.0}, {2.0, 3.0}}, {{4.0, 5.0}, {6.0, 7.0}}};
    Tensor<double, 1> y = {5., -7.};
    Tensor<double, 2> z = {{4, 3}, {2.5, 1.5}};
    Tensor<double, 2> y_z = z * y;
    Tensor<double, 3> w = (x + y).matmul(y_z) * (x + z);
    Tensor<double, 3> dx = w.gradient(x);
    CHECK_EQ(61., dx[0][0][0]);
    CHECK_EQ(-42.5, dx[0][0][1]);
    CHECK_EQ(85.5, dx[0][1][0]);
    CHECK_EQ(-96.0, dx[0][1][1]);
    CHECK_EQ(147., dx[1][0][0]);
    CHECK_EQ(-152, dx[1][0][1]);
    CHECK_EQ(211.5, dx[1][1][0]);
    CHECK_EQ(-214., dx[1][1][1]);
    Tensor<double, 1> dy = w.gradient(y);
    CHECK_EQ(743., dy[0]);
    CHECK_EQ(638.5, dy[1]);
    Tensor<double, 2> dz = w.gradient(z);
    CHECK_EQ(1335., dz[0][0]);
    CHECK_EQ(-1778., dz[0][1]);
    CHECK_EQ(-10., dz[1][0]);
    CHECK_EQ(70., dz[1][1]);
  }
  TEST_CASE("Sub, Mul, Div") {
    Flint::setLoggingLevel(3);
    Tensor<double, 3> u = {{{1, 1}, {2, 2}}, {{3, 3}, {-1, -1}}};
    Tensor<double, 3> x = {{{1.0, 1.0}, {2.0, 3.0}}, {{4.0, 5.0}, {6.0, 7.0}}};
    Tensor<double, 1> y = {5., -7.};
    Tensor<double, 2> z = {{4, 3}, {2.5, 1.5}};
    Tensor<double, 2> y_z = z * y;
    Tensor<double, 3> w = (x - y) / y_z * (x - z) - y_z / z;
    std::cout << (std::string)w << std::endl;
    Tensor<double, 3> dx = w.gradient(x);
    dx.execute();
    std::cout << (std::string)dx << std::endl;
    Tensor<double, 1> dy = w.gradient(y);
    dy.execute();
    std::cout << (std::string)dy << std::endl;
    Tensor<double, 2> dz = w.gradient(z);
    dz.execute();
    std::cout << (std::string)dz << std::endl;
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
