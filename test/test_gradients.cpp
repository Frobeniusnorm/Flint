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
    Tensor<double, 3> x = {{{1.0, 1.0}, {2.0, 3.0}}, {{4.0, 5.0}, {6.0, 7.0}}};
    Tensor<double, 1> y = {5., -7.};
    Tensor<double, 2> z = {{4, 3}, {2.5, 1.5}};
    Tensor<double, 3> w = (x - y) / (z * y) * (x - z) - (z * y);
    Tensor<double, 3> dx = w.gradient(x);
    Tensor<double, 1> dy = w.gradient(y);
    Tensor<double, 2> dz = w.gradient(z);
    using doctest::Approx;
    CHECK_EQ(Approx(-0.35).epsilon(0.001), dx[0][0][0]);
    CHECK_EQ(Approx(-0.28).epsilon(0.001), dx[0][1][0]);
    CHECK_EQ(Approx(-0.05).epsilon(0.001), dx[1][0][0]);
    CHECK_EQ(Approx(-1.8571429).epsilon(0.001), dx[1][1][1]);
    CHECK_EQ(Approx(-13.29).epsilon(0.001), dy[0]);
    CHECK_EQ(Approx(-9.639456).epsilon(0.001), dy[1]);
    CHECK_EQ(Approx(-9.9).epsilon(0.001), dz[0][0]);
    CHECK_EQ(Approx(15.079366).epsilon(0.001), dz[0][1]);
    CHECK_EQ(Approx(-10).epsilon(0.001), dz[1][0]);
    CHECK_EQ(Approx(22.126986).epsilon(0.001), dz[1][1]);
  }
  TEST_CASE("Pow, Neg, Log") {
    Tensor<double, 3> x = {{{42, 75.3}, {4, 4}, {50, 3}},
                           {{7, 9}, {3.5, 77}, {10, 10}}};
    Tensor<double, 1> y = {-7, 5.5};
    Tensor<double, 2> z = {{1.5, 2.5}, {3.5, 4.5}, {7.5, 9}};
    Tensor<double, 3> w = x.pow(y).log();
    Tensor<double, 3> dx = w.gradient(x);
    Tensor<double, 1> dy = w.gradient(y);
    using doctest::Approx;
    CHECK_EQ(Approx(-0.1666666).epsilon(0.001), dx[0][0][0]);
    CHECK_EQ(Approx(0.07304117).epsilon(0.001), dx[0][0][1]);
    CHECK_EQ(Approx(-1.75).epsilon(0.001), dx[0][1][0]);
    CHECK_EQ(Approx(1.833333).epsilon(0.001), dx[0][2][1]);
    CHECK_EQ(Approx(-1).epsilon(0.001), dx[1][0][0]);
    CHECK_EQ(Approx(0.071428).epsilon(0.001), dx[1][1][1]);
    CHECK_EQ(Approx(0.55).epsilon(0.001), dx[1][2][1]);
    CHECK_EQ(Approx(14.537247).epsilon(0.001), dy[0]);
    CHECK_EQ(Approx(15.650002).epsilon(0.001), dy[1]);
    // TODO: test with negative values with integer y
    Tensor<double, 4> t = {{{{-0.5, 3}, {1.5, -1}},
                            {{-3, -2.5}, {1.5, 2.5}},
                            {{-42, -75.3}, {4, -4}}}};
    Tensor<int, 2> r = {{2, 3}, {4, 5}};
    Tensor<double, 4> e = t.pow(r + 1);

    Tensor<double, 4> dt = e.gradient(t);
    CHECK_EQ(Approx(0.75).epsilon(0.001), dt[0][0][0][0]);
    CHECK_EQ(Approx(108).epsilon(0.001), dt[0][0][0][1]);
    CHECK_EQ(Approx(-6).epsilon(0.001), dt[0][0][1][1]);
    CHECK_EQ(Approx(25.312498).epsilon(0.001), dt[0][1][1][0]);
    CHECK_EQ(Approx(585.93744).epsilon(0.01), dt[0][1][1][1]);
    CHECK_EQ(Approx(-1707830.5).epsilon(1), dt[0][2][0][1]);
    CHECK_EQ(Approx(1280.).epsilon(0.01), dt[0][2][1][0]);
    e.execute();
    Tensor<double, 2> dr = e.gradient(r);
    CHECK_EQ(0, dr[0][0]);
    CHECK_EQ(Approx(88.987595).epsilon(0.001), dr[0][1]);
    CHECK_EQ(Approx(1425.7234).epsilon(0.01), dr[1][0]);
    CHECK_EQ(Approx(223.70378).epsilon(0.01), dr[1][1]);
    std::cout << dr << std::endl;
    // TODO: test log2 and log10
  }
}
int main(int argc, char **argv) {
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  flintInit(1, 0);
  Flint::setLoggingLevel(3);
  int res = context.run();
  flintCleanup();
  return res;
}
