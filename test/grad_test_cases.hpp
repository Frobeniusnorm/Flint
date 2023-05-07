#include "../flint.h"
#include "../flint.hpp"
#include "doctest.h"
#include <cmath>
TEST_SUITE("Autodiff") {
  TEST_CASE("Two Times Matmul") {
    Tensor<double, 3> x = {{{1.0, 1.0}, {2.0, 3.0}}, {{4.0, 5.0}, {6.0, 7.0}}};
    x.watch();
    Tensor<double, 2> y = {{3.0, -7.0}, {-1.0, 5.0}};
    y.watch();
    Tensor<double, 3> z = {{{1, 1}, {2, 2}}, {{3, 3}, {-1, -1}}};
    z.watch();
    Tensor<double, 3> w = (x.matmul(y)).matmul(z);
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
    x.watch();
    Tensor<double, 1> y = {5., -7.};
    y.watch();
    Tensor<double, 2> z = {{4, 3}, {2.5, 1.5}};
    z.watch();
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
    Tensor<double, 3> x = {{{1.0, 1.0}, {2.0, 3.0}}, {{4.0, 5.0}, {6.0, 7.0}}};
    Tensor<double, 1> y = {5., -7.};
    Tensor<double, 2> z = {{4, 3}, {2.5, 1.5}};
    x.watch();
    y.watch();
    z.watch();
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
    x.watch();
    y.watch();
    z.watch();
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
    Tensor<float, 4> t = {{{{-0.5, 3}, {1.5, -1}},
                           {{-3, -2.5}, {1.5, 2.5}},
                           {{-42, -75.3}, {4, -4}}}};
    Tensor<int, 2> r = {{2, 3}, {4, 5}};
    t.watch();
    r.watch();
    Tensor<float, 4> e = t.pow(r + 1);

    Tensor<double, 4> dt = e.gradient(t);
    CHECK_EQ(Approx(0.75).epsilon(0.001), dt[0][0][0][0]);
    CHECK_EQ(Approx(108).epsilon(0.001), dt[0][0][0][1]);
    CHECK_EQ(Approx(-6).epsilon(0.001), dt[0][0][1][1]);
    CHECK_EQ(Approx(25.312498).epsilon(0.001), dt[0][1][1][0]);
    CHECK_EQ(Approx(585.93744).epsilon(0.01), dt[0][1][1][1]);
    CHECK_EQ(Approx(-1707830.5).epsilon(1), dt[0][2][0][1]);
    CHECK_EQ(Approx(1280.).epsilon(0.01), dt[0][2][1][0]);
    Tensor<double, 2> dr = e.gradient(r);
    CHECK_EQ(0, dr[0][0]);
    CHECK_EQ(Approx(88.987595).epsilon(0.001), dr[0][1]);
    CHECK_EQ(Approx(1425.7234).epsilon(0.01), dr[1][0]);
    CHECK_EQ(Approx(223.70378).epsilon(0.01), dr[1][1]);
    // test log2 and log10
    Tensor<double, 3> n = x.log10() * (z.log2() + 3);
    Tensor<double, 2> dz = n.gradient(z);
    CHECK_EQ(Approx(2.374048).epsilon(0.001), dz[0][0]);
    CHECK_EQ(Approx(1.633729).epsilon(0.001), dz[0][1]);
    CHECK_EQ(Approx(0.472432).epsilon(0.001), dz[1][0]);
    CHECK_EQ(Approx(0.797826).epsilon(0.001), dz[1][1]);
    CHECK_EQ(Approx(0.519172).epsilon(0.001), dz[2][0]);
    CHECK_EQ(Approx(0.236782).epsilon(0.001), dz[2][1]);
    dx = n.gradient(x);
    CHECK_EQ(Approx(0.037069).epsilon(0.001), dx[0][0][0]);
    CHECK_EQ(Approx(0.024927).epsilon(0.001), dx[0][0][1]);
    CHECK_EQ(Approx(0.521952).epsilon(0.001), dx[0][1][0]);
    CHECK_EQ(Approx(0.893188).epsilon(0.001), dx[0][2][1]);
    CHECK_EQ(Approx(0.222419).epsilon(0.001), dx[1][0][0]);
    CHECK_EQ(Approx(0.029159).epsilon(0.001), dx[1][1][1]);
    CHECK_EQ(Approx(0.256533).epsilon(0.001), dx[1][2][0]);
  }
  TEST_CASE("Min, Max, Abs") {
    Tensor<double, 3> x = {{{42, 75.3}, {4, 4}, {50, 3}},
                           {{7, 9}, {3.5, 77}, {10, 10}}};
    Tensor<double, 1> y = {-7, 5.5};
    Tensor<double, 2> z = {{1.5, 5.5}, {-7, 4.5}, {7.5, -9}};
    x.watch();
    y.watch();
    z.watch();
    Tensor<double, 2> m1 = (z.min(y) * 0.3).abs();
    Tensor<double, 3> m2 = (y.min(z) * 0.3).max(x).abs() * y.abs();
    Tensor<double, 3> dx2 = m2.gradient(x);
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        CHECK_EQ(7, dx2[i][j][0]);
        CHECK_EQ(5.5, dx2[i][j][1]);
      }
    }
    Tensor<double, 1> dy1 = m1.gradient(y);
    CHECK_EQ(-0.6, dy1[0]);
    CHECK_EQ(0, dy1[1]);
    Tensor<double, 1> dy2 = m2.gradient(y);
    CHECK_EQ(-116.5, dy2[0]);
    CHECK_EQ(178.3, dy2[1]);
    Tensor<double, 2> dz1 = m1.gradient(z);
    CHECK_EQ(0, dz1[0][0]);
    CHECK_EQ(0.3, dz1[0][1]);
    CHECK_EQ(-0.3, dz1[1][0]);
    CHECK_EQ(0.3, dz1[1][1]);
    CHECK_EQ(0., dz1[2][0]);
    CHECK_EQ(-0.3, dz1[2][1]);
    Tensor<double, 2> dz2 = m2.gradient(z);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 2; j++)
        CHECK_EQ(0, dz2[i][j]);
  }
  TEST_CASE("Reduce Operations") {
    Tensor<float, 2> a{{0, 3, -1}, {0.5, 2.5, 1}};
    a.watch();
    Tensor<float, 1> b = a.reduce_sum(1) * 2;
    Tensor<double, 2> da = b.gradient(a);
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        CHECK_EQ(2, da[i][j]);
    Tensor<double, 3> x = {{{42, 75.3}, {4, 4}, {50, 3}},
                           {{7, 9}, {3.5, 77}, {10, 10}}};
    x.watch();
    Tensor<double, 1> w = (x.reduce_sum(2) * a).reduce_sum(0);
    da = w.gradient(a);
    Tensor<double, 3> dx = w.gradient(x);
    CHECK_EQ(117.3, da[0][0]);
    CHECK_EQ(8, da[0][1]);
    CHECK_EQ(53, da[0][2]);
    CHECK_EQ(16, da[1][0]);
    CHECK_EQ(80.5, da[1][1]);
    CHECK_EQ(20, da[1][2]);
    CHECK_EQ(0.0, dx[0][0][0]);
    CHECK_EQ(0.0, dx[0][0][1]);
    CHECK_EQ(3.0, dx[0][1][0]);
    CHECK_EQ(3.0, dx[0][1][1]);
    CHECK_EQ(-1, dx[0][2][0]);
    CHECK_EQ(-1, dx[0][2][1]);
    CHECK_EQ(0.5, dx[1][0][0]);
    CHECK_EQ(0.5, dx[1][0][1]);
    CHECK_EQ(2.5, dx[1][1][0]);
    CHECK_EQ(2.5, dx[1][1][1]);
    CHECK_EQ(1, dx[1][2][0]);
    CHECK_EQ(1, dx[1][2][1]);
    Tensor<double, 2> t = (x.reduce_mul(2) * a + 3) * a.reduce_mul(0);
    da = t.gradient(a);
    CHECK_EQ(18.75, da[0][0]);
    CHECK_EQ(-194, da[0][2]);
    CHECK_EQ(0, da[1][0]);
    CHECK_EQ(4204.5, da[1][1]);
    CHECK_EQ(-56, da[1][2]);
    dx = t.gradient(x);
    CHECK_EQ(0, dx[0][0][0]);
    CHECK_EQ(0, dx[0][0][1]);
    CHECK_EQ(90, dx[0][1][0]);
    CHECK_EQ(90, dx[0][1][1]);
    CHECK_EQ(3, dx[0][2][0]);
    CHECK_EQ(50, dx[0][2][1]);
    CHECK_EQ(0, dx[1][0][0]);
    CHECK_EQ(0, dx[1][0][1]);
    CHECK_EQ(1443.75, dx[1][1][0]);
    CHECK_EQ(65.625, dx[1][1][1]);
    CHECK_EQ(-10, dx[1][2][0]);
    CHECK_EQ(-10, dx[1][2][1]);
  }
  TEST_CASE("REPEAT, SLICE, TRANSPOSE") {
    Tensor<double, 2> t{
        {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 0, 1}, {2, 3, 4, 5}, {6, 7, 8, 9}};
    t.watch();
    Tensor<double, 2> r =
        (t.slice(TensorRange(0, 4, 2), TensorRange(-1, -5, -1))).transpose() *
        Tensor<double, 2>{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    Tensor<double, 2> gr = r.gradient(t);
    for (int i = 0; i < 4; i++) {
      CHECK_EQ(0, gr[1][i]);
      CHECK_EQ(0, gr[3][i]);
      CHECK_EQ(0, gr[4][i]);
    }
    CHECK_EQ(8, gr[2][0]);
    CHECK_EQ(7, gr[0][0]);
    CHECK_EQ(6, gr[2][1]);
    CHECK_EQ(5, gr[0][1]);
    CHECK_EQ(4, gr[2][2]);
    CHECK_EQ(3, gr[0][2]);
    CHECK_EQ(2, gr[2][3]);
    CHECK_EQ(1, gr[0][3]);
    gr = t.slice(TensorRange(-1, -2, -1)).repeat(1, 0) *
         Tensor<double, 1>{1, 2, 3, 4};

    gr = gr.gradient(t);
    for (int i = 0; i < 4; i++) {
      CHECK_EQ(0, gr[0][i]);
      CHECK_EQ(0, gr[1][i]);
      CHECK_EQ(0, gr[2][i]);
      CHECK_EQ(0, gr[3][i]);
      CHECK_EQ((i + 1) * 2, gr[4][i]);
    }
  }
  TEST_CASE("SQRT") {
    Tensor<long, 1> y = {9, 7, 13};
    y.watch();
    Tensor<float, 1> z = (y * 0.5f).sqrt();
    Tensor<double, 1> dy = z.gradient(y);
    using doctest::Approx;
    CHECK_EQ(Approx(0.11785114).epsilon(0.000001), dy[0]);
    CHECK_EQ(Approx(0.13363062).epsilon(0.000001), dy[1]);
    CHECK_EQ(Approx(0.09805807).epsilon(0.000001), dy[2]);
  }
  TEST_CASE("SIN, COS, TAN") {
    Tensor<int, 2> x = {{0, 1, -2}, {2, -3, 4}};
    Tensor<long, 1> y = {-9, 7, 13};
    x.watch();
    y.watch();
    Tensor<double, 2> z1 = (x.sin() * y.cos()).tan();
    Tensor<double, 2> dx = z1.gradient(x);
    std::vector<double> res = {-0.91113025, 0.6279001,  -0.8204005,
                               0.8297475,   -0.7548697, -0.99188167};
    dx.execute();
    using doctest::Approx;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        CHECK_EQ(Approx(res[i * 3 + j]).epsilon(0.001), dx[i][j]);
    Tensor<double, 1> dy = z1.gradient(y);
    res = {0.8200625, -0.75841457, 1.3617588};
    for (int j = 0; j < 3; j++)
      CHECK_EQ(Approx(res[j]).epsilon(0.001), dy[j]);
    Tensor<double, 2> z2 = (x.cos().asin() * y.tan().acos()).atan();
    dx = z2.gradient(x);
    res = {0, -0.4722158, 0.89395535, -0.9002461, 0.3335778, 0.67989904};
    CHECK(std::isnan(dx[0][0]));
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++) {
        if (i == 0 && j == 0)
          continue;
        CHECK_EQ(Approx(res[i * 3 + j]).epsilon(0.001), dx[i][j]);
      }
    dy = z2.gradient(y);
    res = {-0.05746716, 1.4498911, 1.0917134};
    for (int j = 0; j < 3; j++)
      CHECK_EQ(Approx(res[j]).epsilon(0.001), dy[j]);
  }
  TEST_CASE("CONVOLVE") {
    Tensor<int, 3> x{{{0, 1, 2},    {1, 2, 3},    {2, 3, 4}},
                     {{3, 4, 5},    {6, 7, 8},    {9, 0, -1}},
                     {{-2, -3, -4}, {-5, -6, -7}, {-8, -9, 0}},
                     {{1, 2, 3},    {4, 5, 6},    {7, 8, 9}}};
    Tensor<int, 3> k{{{1, 1, 1},    {2, 2, 2}}, 
                     {{-3, -3, -3}, {1, 1, 1}}};
    x.watch();
    k.watch();
    Tensor<int, 2> y = x.convolve(k, 1, 2);
    Tensor<double, 3> dk = y.gradient(k);
    CHECK_EQ(12, dk[0][0][0]);
    CHECK_EQ(6, dk[0][0][1]);
    CHECK_EQ(18, dk[0][0][2]);
    CHECK_EQ(6, dk[0][1][0]);
    CHECK_EQ(8, dk[0][1][1]);
    CHECK_EQ(10, dk[0][1][2]);
    CHECK_EQ(10, dk[1][0][0]);
    CHECK_EQ(2, dk[1][0][1]);
    CHECK_EQ(12, dk[1][0][2]);
    CHECK_EQ(5, dk[1][1][0]);
    CHECK_EQ(6, dk[1][1][1]);
    CHECK_EQ(7, dk[1][1][2]);
    Tensor<double, 3> dx = y.gradient(x);
    CHECK_EQ(1, dx[0][0][0]);
    CHECK_EQ(2, dx[0][1][0]);
    CHECK_EQ(1, dx[0][2][0]);
    CHECK_EQ(-2, dx[1][0][0]);
    CHECK_EQ(3, dx[1][1][0]);
    CHECK_EQ(-2, dx[1][2][0]);
    CHECK_EQ(-2, dx[2][0][0]);
    CHECK_EQ(3, dx[2][1][0]);
    CHECK_EQ(-2, dx[2][2][0]);
    CHECK_EQ(-2, dx[3][0][0]);
    CHECK_EQ(3, dx[3][1][0]);
    CHECK_EQ(-2, dx[3][2][0]);
    // check if last dimension is same
    for(int i = 0; i < 4; i++)
      for(int j = 0; j < 3; j++)
	      for(int k = 1; k < 3; k++)
	        CHECK_EQ(dx[i][j][k], dx[i][j][k-1]);
    Tensor<double, 4> w{{{{0.1, 0.2, 0.3}, {-0.9, -0.8, -0.7}}, 
                         {{1, 2, 3}, {0, 0, 0}}}, 
                        {{{3,4,5}, {-1,-1,-1}},
                         {{0, 0, 0}, {1, 2, 0.1}}}};
    Tensor<double, 4> f{{{{3, 2, 1}, {-1, 1, -1}}}};
    w.watch();
    Tensor<double, 3> z = w.convolve(f, 1, 2, 2);
    Tensor<double, 4> dw = z.gradient(w);
    CHECK_EQ(3, dw[0][0][0][0]);
    CHECK_EQ(2, dw[0][0][0][1]);
    CHECK_EQ(1, dw[0][0][0][2]);
    CHECK_EQ(-1, dw[0][0][1][0]);
    CHECK_EQ(1, dw[0][0][1][1]);
    CHECK_EQ(-1, dw[0][0][1][2]);
    CHECK_EQ(0, dw[0][1][0][0]);
    CHECK_EQ(0, dw[0][1][0][1]);
    CHECK_EQ(0, dw[0][1][0][2]);
    CHECK_EQ(0, dw[0][1][1][0]);
    CHECK_EQ(0, dw[0][1][1][1]);
    CHECK_EQ(0, dw[0][1][1][2]);
    CHECK_EQ(3, dw[1][0][0][0]);
    CHECK_EQ(2, dw[1][0][0][1]);
    CHECK_EQ(1, dw[1][0][0][2]);
    CHECK_EQ(-1, dw[1][0][1][0]);
    CHECK_EQ(1, dw[1][0][1][1]);
    CHECK_EQ(-1, dw[1][0][1][2]);
    CHECK_EQ(0, dw[1][1][0][0]);
    CHECK_EQ(0, dw[1][1][0][1]);
    CHECK_EQ(0, dw[1][1][0][2]);
    CHECK_EQ(0, dw[1][1][1][0]);
    CHECK_EQ(0, dw[1][1][1][1]);
    CHECK_EQ(0, dw[1][1][1][2]);
    Tensor<double, 3> a = Tensor<double, 3>::constant(1.0, 6, 6, 1);
    a.watch();
    Tensor<double, 3> b {{{1}, {-1}, {2}, {2}}, {{2}, {3}, {-1}, {4}}};
    Tensor<double, 2> c = a.convolve(b, 5, 2);
    Tensor<double, 3> da = c.gradient(a);
    CHECK_EQ(1, da[0][0][0]);
    CHECK_EQ(-1, da[0][1][0]);
    CHECK_EQ(3, da[0][2][0]);
    CHECK_EQ(1, da[0][3][0]);
    CHECK_EQ(3, da[0][4][0]);
    CHECK_EQ(1, da[0][5][0]);
    CHECK_EQ(2, da[1][0][0]);
    CHECK_EQ(3, da[1][1][0]);
    CHECK_EQ(1, da[1][2][0]);
    CHECK_EQ(7, da[1][3][0]);
    CHECK_EQ(1, da[1][4][0]);
    CHECK_EQ(7, da[1][5][0]);
    for(int i = 0; i < 3; i++)
      for(int j = 0; j < 6; j++)
        CHECK_EQ(0, da[2 + i][j][0]);
    CHECK_EQ(1, da[5][0][0]);
    CHECK_EQ(-1, da[5][1][0]);
    CHECK_EQ(3, da[5][2][0]);
    CHECK_EQ(1, da[5][3][0]);
    CHECK_EQ(3, da[5][4][0]);
    CHECK_EQ(1, da[5][5][0]);

    Tensor<int, 3> s1 = x.slide(k, 1, 2);
    dk = s1.gradient(k);
    CHECK_EQ(s1[0][0][0], dk[0][0][0]);
    CHECK_EQ(s1[0][0][1], dk[0][0][1]);
    CHECK_EQ(s1[0][0][2], dk[0][0][2]);
    CHECK_EQ(10, dk[1][0][0]);
    CHECK_EQ(12, dk[1][0][2]);
    CHECK_EQ(s1[1][1][0], dk[1][1][0]);
    CHECK_EQ(s1[1][1][1], dk[1][1][1]);
    CHECK_EQ(s1[1][1][2], dk[1][1][2]);
    dx = s1.gradient(x);
    for(int i = 0; i < 3; i++){
      CHECK_EQ(1, dx[0][0][i]);
      CHECK_EQ(2, dx[0][1][i]);
      CHECK_EQ(1, dx[0][2][i]);
      for(int j = 1; j < 4; j++){
        CHECK_EQ(-2, dx[j][0][i]);
        CHECK_EQ(3, dx[j][1][i]);
        CHECK_EQ(-2, dx[j][2][i]);
      }
    }
    Tensor<double, 3> dxk = dx.gradient(k);
    for(int i = 0; i < 3; i++){
      CHECK_EQ(8, dxk[0][0][i]);
      CHECK_EQ(4, dxk[0][1][i]);
      CHECK_EQ(6, dxk[1][0][i]);
      CHECK_EQ(3, dxk[1][1][i]);
    }
  }
}
