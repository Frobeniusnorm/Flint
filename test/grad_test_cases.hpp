#include "../flint.h"
#include "../flint.hpp"
#include "doctest.h"
#include <cmath>
#include <math.h>
#include <unordered_map>
#include <unordered_set>
TEST_SUITE("Autodiff") {
	TEST_CASE("Two Times Matmul") {
		GradientContext _;
		Tensor<double, 3> x = {{{1.0, 1.0}, {2.0, 3.0}},
							   {{4.0, 5.0}, {6.0, 7.0}}};
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
		FGraphNode *dxs[3] = {y.get_graph_node(), x.get_graph_node(),
							  z.get_graph_node()};
		FGraphNode *grd[3];
		fCalculateGradients(w.get_graph_node(), &(dxs[0]), 3, &(grd[0]));
		dy = Tensor<double, 2>(grd[0], y.get_shape());
		CHECK_EQ(dy[0][0], 67);
		CHECK_EQ(dy[0][1], 67);
		CHECK_EQ(dy[1][0], 67);
		CHECK_EQ(dy[1][1], 67);
		dx = Tensor<double, 3>(grd[1], x.get_shape());
		CHECK_EQ(dx[0][0][0], 0);
		CHECK_EQ(dx[1][1][1], 0);
		dz = Tensor<double, 3>(grd[2], z.get_shape());
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
		GradientContext _;
		Tensor<double, 3> x = {{{1.0, 1.0}, {2.0, 3.0}},
							   {{4.0, 5.0}, {6.0, 7.0}}};
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
		GradientContext _;
		Tensor<double, 3> x = {{{1.0, 1.0}, {2.0, 3.0}},
							   {{4.0, 5.0}, {6.0, 7.0}}};
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
		GradientContext _;
		Tensor<double, 3> x = {{{42, 75.3}, {4, 4}, {50, 3}},
							   {{7, 9}, {3.5, 77}, {10, 10}}};
		Tensor<double, 1> y = {-7, 5.5};
		Tensor<double, 2> z = {{1.5, 2.5}, {3.5, 4.5}, {7.5, 9}};
		x.watch();
		y.watch();
		z.watch();
		Tensor<double, 3> w = x.pow(y).log();
		FGraphNode *dxs[] = {x.get_graph_node(), y.get_graph_node()};
		FGraphNode *grds[2];
		fCalculateGradients(w.get_graph_node(), &dxs[0], 2, &grds[0]);
		Tensor<double, 3> dx = Tensor<double, 3>(grds[0], x.get_shape());
		Tensor<double, 1> dy = Tensor<double, 1>(grds[1], y.get_shape());
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
		GradientContext _;
		Tensor<double, 3> x = {{{42, 75.3}, {4, 4}, {50, 3}},
							   {{7, 9}, {3.5, 77}, {10, 10}}};
		Tensor<double, 1> y = {-7, 5.5};
		Tensor<double, 2> z = {{1.5, 5.5}, {-7, 4.5}, {7.5, -9}};
		x.watch();
		y.watch();
		z.watch();
		Tensor<double, 2> m1 = (z.min(y) * 0.3).abs();
		Tensor<double, 3> m2 = (y.min(z) * 0.3).max(x).abs() * y.abs();
		FGraphNode *m2dx[] = {x.get_graph_node(), y.get_graph_node(),
							  z.get_graph_node()};
		FGraphNode *m2grds[3];
		fCalculateGradients(m2.get_graph_node(), &m2dx[0], 3, &m2grds[0]);
		Tensor<double, 3> dx2(m2grds[0], x.get_shape());
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 3; j++) {
				CHECK_EQ(7, dx2[i][j][0]);
				CHECK_EQ(5.5, dx2[i][j][1]);
			}
		}
		Tensor<double, 1> dy1 = m1.gradient(y);
		CHECK_EQ(-0.6, dy1[0]);
		CHECK_EQ(0, dy1[1]);
		Tensor<double, 1> dy2(m2grds[1], y.get_shape());
		CHECK_EQ(-116.5, dy2[0]);
		CHECK_EQ(178.3, dy2[1]);
		Tensor<double, 2> dz1 = m1.gradient(z);
		CHECK_EQ(0, dz1[0][0]);
		CHECK_EQ(0.3, dz1[0][1]);
		CHECK_EQ(-0.3, dz1[1][0]);
		CHECK_EQ(0.3, dz1[1][1]);
		CHECK_EQ(0., dz1[2][0]);
		CHECK_EQ(-0.3, dz1[2][1]);
		Tensor<double, 2> dz2(m2grds[2], z.get_shape());
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 2; j++)
				CHECK_EQ(0, dz2[i][j]);
	}
	TEST_CASE("Reduce Operations") {
		GradientContext _;
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
		GradientContext _;
		Tensor<double, 2> t{{0, 1, 2, 3},
							{4, 5, 6, 7},
							{8, 9, 0, 1},
							{2, 3, 4, 5},
							{6, 7, 8, 9}};
		t.watch();
		Tensor<double, 2> r =
			(t.slice(TensorRange(0, 4, 2), TensorRange(-1, -5, -1)))
				.transpose() *
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
		GradientContext _;
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
		GradientContext _;
		Tensor<int, 2> x = {{0, 1, -2}, {2, -3, 4}};
		Tensor<long, 1> y = {-9, 7, 13};
		x.watch();
		y.watch();
		Tensor<double, 2> z1 = (x.sin() * y.cos()).tan();
		Tensor<double, 2> dx = z1.gradient(x);
		std::vector<double> res = {-0.91113025, 0.6279001,	-0.8204005,
								   0.8297475,	-0.7548697, -0.99188167};
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
		GradientContext _;
		Tensor<int, 3> x{{{0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {0, 0, 0}},
						 {{3, 4, 5}, {6, 7, 8}, {9, 0, -1}, {0, 0, 0}},
						 {{-2, -3, -4}, {-5, -6, -7}, {-8, -9, 0}, {0, 0, 0}},
						 {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {0, 0, 0}},
						 {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
		Tensor<int, 3> k{{{1, 1, 1}, {2, 2, 2}}, {{-3, -3, -3}, {1, 1, 1}}};
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
		Tensor<double, 1> m{2, -1.0};
		Tensor<double, 3> dk2 = (y * m).gradient(k);
		Tensor<double, 3> ex2({{{-6., 6., 0.}, {12., 16., 20.}},
							   {{-4., 7., 0.}, {10., 12., 14.}}});
		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++)
				for (int k = 0; k < 2; k++)
					CHECK_EQ(dk2[i][j][k], ex2[i][j][k]);

		Tensor<int, 3> x3{{{0, 1, 2}, {1, 2, 3}, {2, 3, 4}},
						  {{3, 4, 5}, {6, 7, 8}, {9, 0, -1}},
						  {{-2, -3, -4}, {-5, -6, -7}, {-8, -9, 0}},
						  {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
		Tensor<int, 2> y3 = x3.convolve(k, 1, 2);
		Tensor<double, 3> dk3 =
			(y3 * Tensor<double, 2>{{-1}, {2}, {-2}}).gradient(k);
		Tensor<double, 3> ex3{{{10, 13., 16.}, {21., 24., 27.}},
							  {{-9., -14., -19.}, {-24., -29., -34.}}};
		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++)
				for (int k = 0; k < 3; k++)
					CHECK_EQ(dk3[i][j][k], ex3[i][j][k]);

		Tensor<double, 3> dx = y.gradient(x);
		CHECK_EQ(1, dx[0][0][0]);
		CHECK_EQ(2, dx[0][1][0]);
		CHECK_EQ(1, dx[0][2][0]);
		CHECK_EQ(2, dx[0][3][0]);
		CHECK_EQ(-2, dx[1][0][0]);
		CHECK_EQ(3, dx[1][1][0]);
		CHECK_EQ(-2, dx[1][2][0]);
		CHECK_EQ(3, dx[1][3][0]);
		CHECK_EQ(-2, dx[2][0][0]);
		CHECK_EQ(3, dx[2][1][0]);
		CHECK_EQ(-2, dx[2][2][0]);
		CHECK_EQ(3, dx[2][3][0]);
		CHECK_EQ(-2, dx[3][0][0]);
		CHECK_EQ(3, dx[3][1][0]);
		CHECK_EQ(-2, dx[3][2][0]);
		CHECK_EQ(3, dx[3][3][0]);
		CHECK_EQ(-3, dx[4][0][0]);
		CHECK_EQ(1, dx[4][1][0]);
		CHECK_EQ(-3, dx[4][2][0]);
		CHECK_EQ(1, dx[4][3][0]);
		// check if last dimension is same
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 3; j++)
				for (int k = 1; k < 3; k++)
					CHECK_EQ(dx[i][j][k], dx[i][j][k - 1]);
		Tensor<double, 4> w{
			{{{0.1, 0.2, 0.3}, {-0.9, -0.8, -0.7}}, {{1, 2, 3}, {0, 0, 0}}},
			{{{3, 4, 5}, {-1, -1, -1}}, {{0, 0, 0}, {1, 2, 0.1}}}};
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
		Tensor<double, 3> a = Flint::constant(1.0, 6, 6, 1);
		a.watch();
		Tensor<double, 3> b{{{1}, {-1}, {2}, {2}}, {{2}, {3}, {-1}, {4}}};
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
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 6; j++)
				CHECK_EQ(0, da[2 + i][j][0]);

		Tensor<int, 3> s1 = x.slide(k, 1, 2);
		FGraphNode *dxs[] = {k.get_graph_node(), x.get_graph_node()};
		FGraphNode *grads[2];
		fCalculateGradients(s1.get_graph_node(), &dxs[0], 2, &grads[0]);
		dk = Tensor<double, 3>(grads[0], dk.get_shape());
		CHECK_EQ(s1[0][0][0], dk[0][0][0]);
		CHECK_EQ(s1[0][0][1], dk[0][0][1]);
		CHECK_EQ(s1[0][0][2], dk[0][0][2]);
		CHECK_EQ(10, dk[1][0][0]);
		CHECK_EQ(12, dk[1][0][2]);
		CHECK_EQ(s1[1][1][0], dk[1][1][0]);
		CHECK_EQ(s1[1][1][1], dk[1][1][1]);
		CHECK_EQ(s1[1][1][2], dk[1][1][2]);
		dx = Tensor<double, 3>(grads[1], x.get_shape());
		for (int i = 0; i < 3; i++) {
			CHECK_EQ(1, dx[0][0][i]);
			CHECK_EQ(2, dx[0][1][i]);
			CHECK_EQ(1, dx[0][2][i]);
			for (int j = 1; j < 4; j++) {
				CHECK_EQ(-2, dx[j][0][i]);
				CHECK_EQ(3, dx[j][1][i]);
				CHECK_EQ(-2, dx[j][2][i]);
			}
		}
	}
	TEST_CASE("Multifilter Convolve") {
		GradientContext _;
		Tensor<int, 3> x{{{0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {0, 0, 0}},
						 {{3, 4, 5}, {6, 7, 8}, {9, 0, -1}, {0, 0, 0}},
						 {{-2, -3, -4}, {-5, -6, -7}, {-8, -9, 0}, {0, 0, 0}},
						 {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {0, 0, 0}},
						 {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
		Tensor<int, 4> k{{{{1, 1, 1}, {2, 2, 2}}, {{-3, -3, -3}, {1, 1, 1}}},
						 {{{1, 1, 1}, {2, 2, 2}}, {{-3, -3, -3}, {1, 1, 1}}}};
		x.watch();
		k.watch();
		Tensor<int, 3> y = x.convolve(k, 1, 2);
		Tensor<double, 4> dk = y.gradient(k);
		for (int i = 0; i < 2; i++) {
			CHECK_EQ(12, dk[i][0][0][0]);
			CHECK_EQ(6, dk[i][0][0][1]);
			CHECK_EQ(18, dk[i][0][0][2]);
			CHECK_EQ(6, dk[i][0][1][0]);
			CHECK_EQ(8, dk[i][0][1][1]);
			CHECK_EQ(10, dk[i][0][1][2]);
			CHECK_EQ(10, dk[i][1][0][0]);
			CHECK_EQ(2, dk[i][1][0][1]);
			CHECK_EQ(12, dk[i][1][0][2]);
			CHECK_EQ(5, dk[i][1][1][0]);
			CHECK_EQ(6, dk[i][1][1][1]);
			CHECK_EQ(7, dk[i][1][1][2]);
		}
		Tensor<double, 3> dx = y.gradient(x);
		CHECK_EQ(2 * 1, dx[0][0][0]);
		CHECK_EQ(2 * 2, dx[0][1][0]);
		CHECK_EQ(2 * 1, dx[0][2][0]);
		CHECK_EQ(2 * 2, dx[0][3][0]);
		CHECK_EQ(2 * -2, dx[1][0][0]);
		CHECK_EQ(2 * 3, dx[1][1][0]);
		CHECK_EQ(2 * -2, dx[1][2][0]);
		CHECK_EQ(2 * 3, dx[1][3][0]);
		CHECK_EQ(2 * -2, dx[2][0][0]);
		CHECK_EQ(2 * 3, dx[2][1][0]);
		CHECK_EQ(2 * -2, dx[2][2][0]);
		CHECK_EQ(2 * 3, dx[2][3][0]);
		CHECK_EQ(2 * -2, dx[3][0][0]);
		CHECK_EQ(2 * 3, dx[3][1][0]);
		CHECK_EQ(2 * -2, dx[3][2][0]);
		CHECK_EQ(2 * 3, dx[3][3][0]);
		CHECK_EQ(2 * -3, dx[4][0][0]);
		CHECK_EQ(2 * 1, dx[4][1][0]);
		CHECK_EQ(2 * -3, dx[4][2][0]);
		CHECK_EQ(2 * 1, dx[4][3][0]);
		Tensor<float, 4> k2{
			{{{1, 1, 1}, {2, 1, 2}}, {{-3, -3, 3}, {1, 0.5f, 1}}},
			{{{-1, 1, 3}, {0, 4, 1}}, {{-1, 1, 0}, {3, 2, 1}}}};
		k2.watch();
		Tensor<float, 3> y2 =
			x.convolve(k2, 2, 1) * Tensor<float, 3>{{{1, 2}, {-1, 1}, {2, -1}},
													{{3, 2}, {-1, 3}, {1, 1}}};
		Tensor<double, 4> dk2 = y2.gradient(k2);
		Tensor<double, 4> exp = {
			{{{0 * 1 + 1 * -1 + 2 * 2 - 2 * 3 - 5 * -1 - 8 * 1,
			   1 * 1 + 2 * -1 + 3 * 2 - 3 * 3 - 6 * -1 - 9 * 1,
			   2 * 1 + 3 * -1 + 4 * 2 - 4 * 3 - 7 * -1 + 0},
			  {1 * 1 + 2 * -1 + 0 - 5 * 3 - 8 * -1 + 0,
			   2 * 1 + 3 * -1 + 0 - 6 * 3 - 9 * -1 + 0,
			   3 * 1 + 4 * -1 + 0 - 7 * 3 + 0 + 0}},
			 {{3 * 1 + 6 * -1 + 9 * 2 + 1 * 3 + 4 * -1 + 7 * 1,
			   4 * 1 + 7 * -1 + 0 + 2 * 3 + 5 * -1 + 8 * 1,
			   5 * 1 + 8 * -1 - 1 * 2 + 3 * 3 + 6 * -1 + 9 * 1},
			  {6 * 1 + 9 * -1 + 4 * 3 + 7 * -1, 7 * 1 + 0 * -1 + 5 * 3 + 8 * -1,
			   8 * 1 - 1 * -1 + 6 * 3 + 9 * -1}}},
			{{{0 * 2 + 1 * 1 + 2 * -1 - 2 * 2 - 5 * 3 - 8 * 1,
			   1 * 2 + 2 * 1 + 3 * -1 - 3 * 2 - 6 * 3 - 9 * 1,
			   2 * 2 + 3 * 1 + 4 * -1 - 4 * 2 - 7 * 3 + 0},
			  {1 * 2 + 2 * 1 + 0 - 5 * 2 - 8 * 3 + 0,
			   2 * 2 + 3 * 1 + 0 - 6 * 2 - 9 * 3 + 0,
			   3 * 2 + 4 * 1 + 0 - 7 * 2 + 0 + 0}},
			 {{3 * 2 + 6 * 1 + 9 * -1 + 1 * 2 + 4 * 3 + 7 * 1,
			   4 * 2 + 7 * 1 + 0 + 2 * 2 + 5 * 3 + 8 * 1,
			   5 * 2 + 8 * 1 - 1 * -1 + 3 * 2 + 6 * 3 + 9 * 1},
			  {6 * 2 + 9 * 1 + 4 * 2 + 7 * 3, 7 * 2 + 0 * 1 + 5 * 2 + 8 * 3,
			   8 * 2 - 1 * 1 + 6 * 2 + 9 * 3}}}};
		for (int i = 0; i < exp.get_shape()[0]; i++) {
			for (int j = 0; j < exp.get_shape()[1]; j++) {
				for (int k = 0; k < exp.get_shape()[2]; k++) {
					for (int l = 0; l < exp.get_shape()[3]; l++) {
						CHECK_EQ(exp[i][j][k][l], dk2[i][j][k][l]);
					}
				}
			}
		}
	}
	TEST_CASE("Concat, Exponential") {
		GradientContext _;
		Tensor<int, 2> a{{0, 1}, {2, 3}};
		a.watch();
		Tensor<int, 2> b{{4, 5}, {6, 7}};
		Tensor<double, 1> e{4.2, -6, 7, 4};
		Tensor<double, 2> c = (Flint::concat(a, b, 1) * e);
		Tensor<double, 2> da = c.gradient(a);
		CHECK_EQ(doctest::Approx((4.2)), da[0][0]);
		CHECK_EQ(doctest::Approx((-6)), da[0][1]);
		CHECK_EQ(doctest::Approx((4.2)), da[1][0]);
		CHECK_EQ(doctest::Approx((-6)), da[1][1]);
		e.watch();
		Tensor<double, 1> eexp = e.exp() * 2;
		Tensor<double, 1> de = eexp.gradient(e);
		Tensor<double, 1> dec = (e * 2).exp();
		for (int i = 0; i < 4; i++)
			CHECK_EQ(dec[i], dec[i]);
	}
	TEST_CASE("Index, Set Index") {
		GradientContext _;
		Tensor<double, 3> a = {
			{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}, {{8, 9}, {10, 11}}};
		a.watch();
		Tensor<int, 1> i1 = {0, 2};
		Tensor<double, 3> a1 =
			a.index(i1) *
			Tensor<double, 3>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
		Tensor<double, 3> ga1 = a1.gradient(a);
		Tensor<double, 3> e1 = {
			{{1, 2}, {3, 4}}, {{0, 0}, {0, 0}}, {{5, 6}, {7, 8}}};
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 2; j++)
				for (int k = 0; k < 2; k++)
					CHECK_EQ(e1[i][j][k], ga1[i][j][k]);
		Tensor<int, 2> i2 = {{0, 0, 1, 1}, {1, 0, 1, 0}, {0, 1, 1, 0}};
		Tensor<double, 3> a2 =
			a.index(i2) * Tensor<double, 3>{{{1, 2}, {3, 4}, {5, 6}, {7, 8}},
											{{9, 1}, {2, 3}, {4, 5}, {6, 7}},
											{{8, 9}, {0, 1}, {2, 3}, {4, 5}}};
		Tensor<double, 3> e2 = {{{4, 6}, {12, 14}},
								{{2 + 6, 3 + 7}, {9 + 4, 1 + 5}},
								{{12, 14}, {2, 4}}};
		Tensor<double, 3> g2 = a2.gradient(a);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 2; j++)
				for (int k = 0; k < 2; k++)
					CHECK_EQ(e2[i][j][k], g2[i][j][k]);

		Tensor<double, 3> a3 = Flint::random(3, 3, 3);
		Tensor<double, 3> b3 = Flint::random(3, 3, 3);
		a3.watch();
		b3.watch();
		Tensor<int, 1> i3 = {0, 0, 2};
		Tensor<double, 3> m3 = Flint::random(3, 3, 3);
		Tensor<double, 3> c3 = a3.index_set(b3, i3) * m3;
		Tensor<double, 3> g3 = c3.gradient(a3);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				for (int k = 0; k < 3; k++) {
					if (i == 1)
						CHECK_EQ(m3[i][j][k], g3[i][j][k]);
					else
						CHECK_EQ(0, g3[i][j][k]);
				}
	}
	TEST_CASE("Reduce_min/max") {
		GradientContext _;
		Tensor<int, 3> a{{{0, 9, 4}, {-1, 7, 4}, {7, 7, 2}}};
		a.watch();
		Tensor<float, 2> a1 = a.reduce_max(0) * 42.0f;
		Tensor<double, 3> da1 = a1.gradient(a);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++) {
				CHECK_EQ(doctest::Approx(42.0), da1[0][i][j]);
			}
		Tensor<float, 2> a2 = a.reduce_max(1) * 42.0f;
		Tensor<double, 3> da2 = a2.gradient(a);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++) {
				if ((i == 0 && j == 1) || ((i == 0 || i == 1) && j == 2) ||
					(i == 2 && j == 0))
					CHECK_EQ(doctest::Approx(42.0), da2[0][i][j]);
				else
					CHECK_EQ(doctest::Approx(0.0), da2[0][i][j]);
			}
		Tensor<float, 2> a3 = a.reduce_max(2) * 42.0f;
		Tensor<double, 3> da3 = a3.gradient(a);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++) {
				if ((i == 0 && j == 1) || (i == 1 && j == 1) ||
					(i == 2 && (j == 0 || j == 1)))
					CHECK_EQ(doctest::Approx(42.0), da3[0][i][j]);
				else
					CHECK_EQ(doctest::Approx(0.0), da3[0][i][j]);
			}
		Tensor<float, 3> b{{{0.1234, 9.7152, 4.1111},
							{-1.1111, 7.42313574159321333, 4.1111},
							{7.42313574159321333, 7.42313574159321333, 2}}};
		b.watch();
		Tensor<float, 2> b4 = b.reduce_max(2) * 42.0f;
		Tensor<double, 3> db4 = b4.gradient(b);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++) {
				if ((i == 0 && j == 1) || (i == 1 && j == 1) ||
					(i == 2 && (j == 0 || j == 1)))
					CHECK_EQ(doctest::Approx(42.0), db4[0][i][j]);
				else
					CHECK_EQ(doctest::Approx(0.0), db4[0][i][j]);
			}
	}
	TEST_CASE("Sliding Window") {
		GradientContext _;
		Tensor<double, 2> a1 = {
			{0, 1, 2, 3}, {10, 11, 12, 13}, {20, 21, 22, 23}};
		Tensor<double, 1> b1 = {1, -1, 2, -2, 3, -3};
		Tensor<double, 2> e1 = {{1, 0, 1, 2}, {-1, 1, 1, -1}, {-2, 1, 0, -3}};
		a1.watch();
		Tensor<double, 3> y1 =
			a1.sliding_window(std::array<size_t, 2>{2, 2},
							  std::array<unsigned int, 2>{1, 1}) *
			b1.reshape(6, 1, 1).repeat(0, 1, 1);
		Tensor<double, 2> g1 = y1.gradient(a1);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 4; j++)
				CHECK_EQ(doctest::Approx(g1[i][j]), e1[i][j]);
		Tensor<double, 3> b2 = {{{-1, 1}}, {{2, 3}}, {{3, 4}}, {{5, 6}}};
		Tensor<double, 3> y2 =
			a1.sliding_window(std::array<size_t, 2>{1, 2},
							  std::array<unsigned int, 2>{2, 2}) *
			b2;
		Tensor<double, 2> g2 = y2.gradient(a1);
		Tensor<double, 2> e2 = {{-1, 1, 2, 3}, {0, 0, 0, 0}, {3, 4, 5, 6}};
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 4; j++)
				CHECK_EQ(doctest::Approx(g2[i][j]), e2[i][j]);
	}
	TEST_CASE("Reduce and calculate with itself") {
		GradientContext _;
		Tensor<float, 3> in = {{{77, -3, 76, 79}, {123, 54, 1024, 1023}},
							   {{0.5, 0.9, -312, 2}, {-5, -6, -7, -8}}};
		in.watch();
		Tensor<float, 3> expected = {{{0, 0, 1, 0}, {0, 0, 1, 0}},
									 {{0, 1, 0, 0}, {0, 1, 0, 0}}};
		Tensor<float, 3> pred =
			(in / in.reduce_sum(2).expand(2, in.get_shape()[2]));
		Tensor<double, 3> grad = pred.gradient(in);
		for (int j = 0; j < 4; j++) {
			for (int i = 0; i < 2; i++) {
				CHECK_EQ(doctest::Approx(0), grad[0][i][j]);
			}
			CHECK_EQ(doctest::Approx(2.33e-10), grad[1][0][j]);
			CHECK_EQ(doctest::Approx(3.73e-09), grad[1][1][j]);
		}
	}
	TEST_CASE("Use one variable multiple times") {
		GradientContext _;
		Tensor<float, 3> in = {{{77, -3, 76, 79}, {123, 54, 1024, 1023}},
							   {{0.5, 0.9, -312, 2}, {-5, -6, -7, -8}}};
		in.watch();
		auto v = in * 7;
		auto t1 = v / v;
		auto t2 = v / (in * 7);
		auto g1 = t1.gradient(in);
		auto g2 = t2.gradient(in);
		CHECK_EQ((t1.equal(t2) - 1).reduce_sum()[0], 0);
		CHECK_EQ((g1.equal(g2) - 1).reduce_sum()[0], 0);
		auto t3 = in * 7;
		t3 = t3 / (in * 7).reduce_sum(2).expand(2, 4);
		auto t4 = in * 7;
		t4 = t4 / t4.reduce_sum(2).expand(2, 4);
		auto g3 = t3.gradient(in);
		auto g4 = t4.gradient(in);
		CHECK_EQ((t3.equal(t4) - 1).reduce_sum()[0], 0);
		CHECK_EQ(((g3 - g4).abs() > 0.0001).reduce_sum()[0], 0);
	}
	TEST_CASE("Pooling") {
		GradientContext _;
		Tensor<int, 3> x1{{{0, 1, 2}, {1, 2, 3}, {2, 3, 4}},
						  {{3, 4, 5}, {6, 7, 8}, {9, 0, -1}},
						  {{-2, -3, -4}, {-5, -6, -7}, {-8, -9, 0}},
						  {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
		x1.watch();
		Tensor<double, 2> c1{{2, -1}, {1, 3}, {3, -2}};
		Tensor<double, 2> y1 = x1.pooling_sum({2, 2}, {1, 1}).convert<double>();
		y1 = y1 * c1;
		Tensor<double, 3> dx1 = y1.gradient(x1);
		Tensor<double, 3> ex1 = Tensor<double, 3>{
			{{2}, {1}, {-1}},
			{{3}, {5}, {2}},
			{{4}, {5}, {1}},
			{{3}, {1}, {-2}}}.repeat(0, 0, 2);
		std::cout << dx1() << std::endl;
		for (int i = 0; i < ex1.get_shape()[0]; i++) {
			for (int j = 0; j < ex1.get_shape()[1]; j++) {
				for (int k = 0; k < ex1.get_shape()[2]; k++) {
					CHECK_EQ(ex1[i][j][k], dx1[i][j][k]);
				}
			}
		}
	}
}
