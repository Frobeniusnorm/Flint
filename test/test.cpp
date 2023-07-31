/* Copyright 2022 David Schwarzbeck

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "../flint.h"
#include <cmath>
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "testutils.hpp"
#include <vector>

#include "../flint.hpp"
TEST_SUITE("Graph implementation") {
  TEST_CASE("Set by index") {
    Tensor<double, 3> a1 = Flint::random(2, 2, 2);
    Tensor<double, 3> b1 = Flint::random(5, 2, 2);
    Tensor<int, 1> i1 = {-1, 1};
    Tensor<double, 3> c1 = a1.set_by_index(b1, i1);
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) {
        CHECK_EQ(doctest::Approx(c1[0][i][j]), a1[0][i][j]);
        CHECK_EQ(doctest::Approx(c1[1][i][j]), b1[1][i][j]);
      }
    Tensor<double, 3> a2 = Flint::random(2, 5, 2)();
    Tensor<double, 3> b2 = Flint::random(2, 3, 2)();
    Tensor<int, 2> i2 = {{-1, 1, 0, -1, 2}, {-1, -1, 2, 1, 0}};
    Tensor<double, 3> c2 = a2.set_by_index(b2, i2)();
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++) {
          CHECK_EQ(doctest::Approx(c2[i][j][k]),
                   i2[i][j] < 0 ? a2[i][j][k] : b2[i][i2[i][j]][k]);
        }
    // multi index set
    Tensor<int, 2> a3 = {{0, 1}, {2, 3}, {4, 5}, {6, 7}}; 
    Tensor<int, 2> b3 = {{4, 5}, {6, 7}, {8, 9}}; 
    Tensor<int, 1> i3 = {0, 0, 2};
    Tensor<int, 2> e3 = {{10, 12}, {2, 3}, {8, 9}, {6, 7}};
    Tensor<int, 2> c3 = a3.multi_index_set(b3, i3);
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 2; j++)
        CHECK_EQ(c3[i][j], e3[i][j]);
    Tensor<int, 2> i4 = {{-1, 0}, {1, 1}, {1, 0}, {1, -1}};
    Tensor<int, 2> b4 = {{4, 5}, {6, 7}, {8, 9}, {10, 11}}; 
    Tensor<int, 2> e4 = {{5, 1}, {2, 13}, {9, 8}, {6, 10}};
    Tensor<int, 2> c4 = a3.multi_index_set(b4, i4);
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 2; j++)
        CHECK_EQ(c4[i][j], e4[i][j]);
    
  }
  TEST_CASE("createGraph, add, mul, sub, div") {
    using namespace std;
    {
      vector<double> v1(100);
      vector<float> v2(100);
      // construct graph 1
      vector<size_t> shape{100};
      FGraphNode *gn1 =
          fCreateGraph(v1.data(), v1.size(), F_FLOAT64, shape.data(), 1);
      gn1 = fadd(gn1, 7.0);
      FGraphNode *gn12 =
          fCreateGraph(v2.data(), v2.size(), F_FLOAT32, shape.data(), 1);
      gn1 = fmul(gn1, gn12);
      fFreeGraph(gn12);
      //  test
      REQUIRE_EQ(gn1->num_predecessor, 2);
      CHECK_EQ(gn1->operation.data_type, F_FLOAT64);
      FGraphNode *right1 = gn1->predecessors[1];
      CHECK_EQ(right1->num_predecessor, 0);
      CHECK(right1->predecessors == nullptr);
      FStore *store1 = (FStore *)right1->operation.additional_data;
      CHECK_EQ(right1->operation.data_type, F_FLOAT32);
      CHECK_EQ(store1->num_entries, 100);
      fFreeGraph(gn1);
    }
    {
      vector<long> v1(100);
      vector<int> v2(100, 1);
      // construct graph 2
      vector<size_t> shape = {10, 10};
      FGraphNode *gn2 =
          fCreateGraph(v1.data(), v1.size(), F_INT64, shape.data(), 2);
      gn2 = fsub(gn2, 7);
      FGraphNode *gn21 =
          fCreateGraph(v2.data(), v2.size(), F_INT32, shape.data(), 2);
      gn2 = fdiv(gn2, gn21);
      // test
      REQUIRE_EQ(gn2->num_predecessor, 2);
      CHECK_EQ(gn2->operation.op_type, FDIV);
      CHECK_EQ(gn2->operation.data_type, F_INT64);
      FGraphNode *right2 = gn2->predecessors[1];
      CHECK_EQ(right2->num_predecessor, 0);
      CHECK(right2->predecessors == nullptr);
      FStore *store2 = (FStore *)right2->operation.additional_data;
      CHECK_EQ(right2->operation.data_type, F_INT32);
      CHECK_EQ(store2->num_entries, 100);
      FGraphNode *left1 = gn2->predecessors[0];
      FGraphNode *const1 = left1->predecessors[1];
      CHECK_EQ(const1->operation.op_type, FSTORE);
      fFreeGraph(gn2);
    }
  }
  TEST_CASE("serialize, unserialize") {
    using namespace std;
    vector<double> v1(6);
    v1[0] = -1.5;
    v1[1] = -1.0;
    v1[2] = -0.5;
    v1[3] = 0;
    v1[4] = 0.5;
    v1[5] = 1.0;
    vector<size_t> shape{2, 3};
    FGraphNode *gn1 =
        fCreateGraph(v1.data(), v1.size(), F_FLOAT64, shape.data(), 2);
    char *data = fserialize(gn1, nullptr);
    fFreeGraph(gn1);
    FGraphNode *gnp2 = fdeserialize(data);
    free(data);
    fCalculateResult(gnp2);
    for (int i = 0; i < 6; i++)
      CHECK_EQ(-1.5 + i * 0.5, ((double *)gnp2->result_data->data)[i]);
    fFreeGraph(gnp2);
  }
}
TEST_SUITE("Execution") {
  TEST_CASE("init, execution (add, sub, mul) and cleanup") {
    using namespace std;
    vector<double> v1(10, 4.0);
    vector<float> v2(10, 4.0f);
    // construct graph 1
    vector<size_t> shape{10};
    FGraphNode *gn1 =
        fCreateGraph(v1.data(), v1.size(), F_FLOAT64, shape.data(), 1);
    gn1 = fadd(gn1, 7.0);
    FGraphNode *gn11 =
        fCreateGraph(v2.data(), v2.size(), F_FLOAT32, shape.data(), 1);
    gn1 = fmul(gn1, gn11);
    fFreeGraph(gn11); // delete handle
    FGraphNode *result = fCalculateResult(gn1);
    FResultData *rd = result->result_data;
    CHECK_EQ(rd->num_entries, 10);
    for (size_t i = 0; i < rd->num_entries; i++)
      CHECK_EQ(((double *)rd->data)[i], 44);
    // construct graph 2 (first not-tree)
    vector<float> v3(10);
    for (int i = 0; i < 10; i++)
      v3[i] = i + 1;
    FGraphNode *gn2 =
        fCreateGraph(v3.data(), v3.size(), F_FLOAT32, shape.data(), 1);
    FGraphNode *gn3 = fadd(gn2, result);
    gn3 = fadd(gn3, result);
    gn3 = fsub(gn3, 80);
    gn3 = fadd(gn3, gn2);
    result = fCalculateResult(gn3);
    rd = result->result_data;
    CHECK_EQ(rd->num_entries, 10);
    for (int i = 0; i < 10; i++)
      CHECK_EQ(((double *)rd->data)[i], 8 + (i + 1) * 2);
    fFreeGraph(result);
  }
  TEST_CASE("Multidimensional Data") {
    using namespace std;
    vector<vector<double>> v1{
        {0.0, 1.0, 2.0}, {0.0, -1.0, -2.0}, {0.0, 1.0, 2.0}};
    vector<vector<double>> v2{
        {2.0, 1.0, 0.0}, {0.0, -1.0, -2.0}, {2.0, 1.0, 2.0}};
    vector<double> f1 = flattened(v1);
    vector<double> f2 = flattened(v2);
    vector<size_t> shape{3, 3};
    FGraphNode *gn1 =
        fCreateGraph(f1.data(), f1.size(), F_FLOAT64, shape.data(), 2);
    FGraphNode *gn2 =
        fCreateGraph(f2.data(), f2.size(), F_FLOAT64, shape.data(), 2);
    FGraphNode *gn3 = fadd(gn1, gn2);
    FGraphNode *result = fCalculateResult(gn3);
    FResultData *rd = result->result_data;
    CHECK_EQ(rd->num_entries, 9);
    REQUIRE_EQ(result->operation.dimensions, 2);
    CHECK_EQ(result->operation.shape[0], 3);
    CHECK_EQ(result->operation.shape[1], 3);
    CHECK_EQ(result->operation.data_type, F_FLOAT64);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        CHECK_EQ(((double *)rd->data)[i * 3 + j], v1[i][j] + v2[i][j]);
    // 3d + 2d
    vector<vector<vector<int>>> v3{{{0, 1, 2}, {2, 1, 0}, {0, 1, 2}},
                                   {{5, 9, 2}, {3, 5, 7}, {3, 4, 1}},
                                   {{0, 1, 2}, {9, 8, 7}, {5, 9, 7}},
                                   {{-3, -2, 4}, {-1, -2, 3}, {11, 1, 0}}};

    vector<int> f3 = flattened(v3);
    vector<size_t> shape_f3{4, 3, 3};
    FGraphNode *gn4 =
        fCreateGraph(f3.data(), f3.size(), F_INT32, shape_f3.data(), 3);
    FGraphNode *gn5 = fadd(gn4, result);
    FGraphNode *newResult;
    for (int i = 0; i < 2; i++) {
      newResult = fCalculateResult(gn5);
      rd = newResult->result_data;

      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
          for (int k = 0; k < 3; k++)
            CHECK_EQ(((double *)rd->data)[i * 9 + j * 3 + k],
                     v1[j][k] + v2[j][k] + v3[i][j][k]);
    }
    fFreeGraph(newResult);
  }
  TEST_CASE("pow") {
    using namespace std;
    vector<size_t> s1{3, 2};
    size_t s2 = 2;
    vector<vector<int>> d1{{1, 3}, {0, 8}, {-3, -3}};
    vector<int> f1 = flattened(d1);
    vector<long> d2{2, 1};
    vector<vector<float>> d3{{0, 2}, {1, 0}, {-1, 2}};
    vector<float> f3 = flattened(d3);
    FGraphNode *g1 = fCreateGraph(f1.data(), f1.size(), F_INT32, s1.data(), 2);
    FGraphNode *g12 = fCreateGraph(d2.data(), d2.size(), F_INT64, &s2, 1);
    FGraphNode *g2 = fpow(g1, g12);
    FGraphNode *g13 =
        fCreateGraph(f3.data(), f3.size(), F_FLOAT32, s1.data(), 2);
    FGraphNode *g3 = fpow(g1, g13);
    fFreeGraph(g13); // delete handles
    fFreeGraph(g12);
    FGraphNode *g4 = fpow(g1, 2);
    vector<vector<long>> e1{{1, 3}, {0, 8}, {9, -3}};
    vector<vector<float>> e2{{1, 9}, {0, 1}, {-0.3333333333333333, 9}};
    vector<vector<int>> e3{{1, 9}, {0, 64}, {9, 9}};

    FGraphNode *r1 = fCalculateResult(g2);
    FGraphNode *r3 = fCalculateResult(g4);
    FGraphNode *r2 = fCalculateResult(g3);
    CHECK_EQ(2, r3->operation.dimensions);
    CHECK_EQ(3, r3->operation.shape[0]);
    FResultData *res = r1->result_data;
    long *ldata = (long *)res->data;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 2; j++)
        CHECK_EQ(ldata[i * 2 + j], e1[i][j]);

    res = r2->result_data;
    float *fdata = (float *)res->data;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 2; j++)
        CHECK_EQ(fdata[i * 2 + j], e2[i][j]);

    res = r3->result_data;
    int *idata = (int *)res->data;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 2; j++)
        CHECK_EQ(idata[i * 2 + j], e3[i][j]);
    fFreeGraph(r1);
    fFreeGraph(r2);
    fFreeGraph(r3);
  }
  TEST_CASE("flatten, reshape") {
    using namespace std;
    vector<vector<int>> d1{{1, 3}, {0, 8}, {-3, -3}};
    vector<int> f1 = flattened(d1);
    vector<int> d2{3, 3, 4, 4, 5, 5};
    vector<int> e1{4, 6, 4, 12, 2, 2};
    vector<size_t> s1{3, 2};
    size_t s2 = 6;
    FGraphNode *g = fCreateGraph(f1.data(), f1.size(), F_INT32, s1.data(), 2);
    FGraphNode *gi = fCreateGraph(d2.data(), d2.size(), F_INT32, &s2, 1);
    g = fadd(fflatten(g), gi);
    fFreeGraph(gi);
    g = fCalculateResult(g);
    FResultData *res = g->result_data;
    int *data = (int *)res->data;
    for (int i = 0; i < 6; i++)
      CHECK_EQ(data[i], e1[i]);
    fFreeGraph(g);
    // flatten with index
    vector<vector<vector<int>>> d3{{{0, 1}, {2, 3}, {4, 5}},
                                   {{6, 7}, {8, 9}, {10, 11}}};
    vector<int> f3 = flattened(d3);
    vector<size_t> s3{2, 3, 2};
    vector<vector<int>> d4{{3, 3, 4, 4, 5, 5}, {5, 5, 4, 4, 3, 3}};
    vector<int> f4 = flattened(d4);
    vector<size_t> s4{2, 6};
    vector<vector<int>> d5{{3, 3}, {4, 4}, {5, 5}, {5, 5}, {4, 4}, {3, 3}};
    vector<int> f5 = flattened(d5);
    vector<size_t> s5{6, 2};
    g = fCreateGraph(f3.data(), f3.size(), F_INT32, s3.data(), 3);
    FGraphNode *g1 = fflatten(g, 2);
    FGraphNode *g2 = fflatten(g, 1);
    FGraphNode *g11 = fCreateGraph(f4.data(), f4.size(), F_INT32, s4.data(), 2);
    FGraphNode *g21 = fCreateGraph(f5.data(), f5.size(), F_INT32, s5.data(), 2);
    g1 = fflatten(fadd(g1, g11));
    g2 = fflatten(fadd(g2, g21));
    fFreeGraph(g11);
    fFreeGraph(g21);
    vector<int> exp{3, 4, 6, 7, 9, 10, 11, 12, 12, 13, 13, 14};
    g1 = fCalculateResult(g1);
    g2 = fCalculateResult(g2);
    int *r1 = (int *)(g1->result_data->data);
    int *r2 = (int *)(g2->result_data->data);
    for (int i = 0; i < 12; i++) {
      CHECK_EQ(r1[i], exp[i]);
      CHECK_EQ(r2[i], exp[i]);
    }
    fFreeGraph(g1);
    fFreeGraph(g2);
    // more complicated
    Tensor<int, 3> t1{{{0, 1}, {2, 3}, {4, 5}}, {{6, 7}, {8, 9}, {10, 11}}};
    Tensor<int, 2> t2{{1, 1}, {1, 1}, {1, 1}};
    Tensor<int, 3> t3 = t1 + t2;
    Tensor<int, 2> t4 = {{11, 10}, {9, 8}, {7, 6}, {5, 4}, {3, 2}, {1, 0}};
    Tensor<int, 2> t5 = t3.flattened(1) + t4;
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 2; j++) {
        CHECK_EQ(i * 2 + j + 1 + (11 - i * 2 - j), t5[i][j]);
      }
    }
    Tensor<int, 4> t6 = t1.reshape(2, 3, 2, 1);
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 2; k++)
          CHECK_EQ(i * 6 + j * 2 + k, t6[i][j][k][0]);
  }
  TEST_CASE("neg") {
    using namespace std;
    vector<float> data1{1, 2, 3, 4};
    vector<float> data2{4, 3, 2, 1};
    vector<size_t> s1{2, 2};
    FGraphNode *g1 =
        fCreateGraph(data1.data(), data1.size(), F_FLOAT32, s1.data(), 2);
    FGraphNode *g2 =
        fCreateGraph(data2.data(), data2.size(), F_FLOAT32, s1.data(), 2);
    g1 = fCalculateResult(fneg(g1));
    g2 = fCalculateResult(fneg(g2));
    FResultData *rd1 = g1->result_data;
    FResultData *rd2 = g2->result_data;
    CHECK_EQ(-1, ((float *)rd1->data)[0]);
    CHECK_EQ(-2, ((float *)rd1->data)[1]);
    CHECK_EQ(-3, ((float *)rd1->data)[2]);
    CHECK_EQ(-4, ((float *)rd1->data)[3]);
    CHECK_EQ(-4, ((float *)rd2->data)[0]);
    CHECK_EQ(-3, ((float *)rd2->data)[1]);
    CHECK_EQ(-2, ((float *)rd2->data)[2]);
    CHECK_EQ(-1, ((float *)rd2->data)[3]);
    fFreeGraph(g1);
    fFreeGraph(g2);
  }
  TEST_CASE("matmul") {
    using namespace std;
    vector<float> data1{1, 2, 3, 4};
    vector<float> data2{4, 3, 2, 1};
    vector<size_t> s1{2, 2};
    FGraphNode *g1 =
        fCreateGraph(data1.data(), data1.size(), F_FLOAT32, s1.data(), 2);
    FGraphNode *g2 =
        fCreateGraph(data2.data(), data2.size(), F_FLOAT32, s1.data(), 2);
    FGraphNode *mm1 = fmatmul(g1, g2);
    FGraphNode *r1 = fCalculateResult(mm1);
    FResultData *rd1 = r1->result_data;
    vector<float> exp1{4 + 4, 3 + 2, 12 + 8, 9 + 4};
    float *d1 = (float *)rd1->data;
    for (int i = 0; i < 4; i++)
      CHECK_EQ(exp1[i], d1[i]);
    fFreeGraph(r1);

    // different sizes along axis
    vector<int> data4{6, 5, 4, 3, 2, 1};
    vector<int> data3{1, 2, 3, 4, 5, 6};
    vector<int> exp2{1 * 6 + 2 * 4 + 3 * 2, 1 * 5 + 2 * 3 + 3 * 1,
                     4 * 6 + 5 * 4 + 6 * 2, 4 * 5 + 5 * 3 + 6 * 1};
    s1 = vector<size_t>{2, 3};
    vector<size_t> s2{3, 2};
    vector<size_t> s3{2, 2};
    g1 = fCreateGraph(data3.data(), data3.size(), F_INT32, s1.data(), 2);
    g2 = fCreateGraph(data4.data(), data4.size(), F_INT32, s2.data(), 2);
    FGraphNode *mm2 = fmatmul(g1, g2);
    REQUIRE_EQ(mm2->operation.shape[0], s3[0]);
    REQUIRE_EQ(mm2->operation.shape[1], s3[1]);
    FGraphNode *r2 = fCalculateResult(mm2);
    FResultData *rd2 = r2->result_data;
    int *d2 = (int *)rd2->data;
    for (int i = 0; i < 4; i++)
      CHECK_EQ(exp2[i], d2[i]);
    fFreeGraph(r2);

    // multidim test
    vector<vector<vector<double>>> data5{{{0, 1, 2}, {1, 2, 3}},
                                         {{2, 3, 4}, {3, 4, 5}}};
    vector<size_t> s5{2, 2, 3};
    vector<double> f5 = flattened(data5);

    vector<vector<float>> data6{{0, 1}, {2, 3}, {4, 5}};
    vector<size_t> s6{3, 2};
    vector<float> f6 = flattened(data6);

    vector<vector<vector<double>>> exp3{{{10, 13}, {16, 22}},
                                        {{22, 31}, {28, 40}}};
    vector<double> fe3 = flattened(exp3);

    g1 = fCreateGraph(f5.data(), f5.size(), F_FLOAT64, s5.data(), s5.size());
    g2 = fCreateGraph(f6.data(), f6.size(), F_FLOAT32, s6.data(), s6.size());
    mm2 = fmatmul(g1, g2);
    REQUIRE_EQ(mm2->operation.shape[0], 2);
    REQUIRE_EQ(mm2->operation.shape[1], 2);
    REQUIRE_EQ(mm2->operation.shape[2], 2);
    r2 = fCalculateResult(mm2);
    FResultData *rd3 = r2->result_data;
    double *d3 = (double *)rd3->data;
    for (size_t i = 0; i < rd3->num_entries; i++)
      CHECK_EQ(fe3[i], d3[i]);
    fFreeGraph(r2);
  }
}
#include <chrono>
TEST_SUITE("C++ Bindings") {
  TEST_CASE("Constant Functions") {
    Tensor<float, 3> t1 = Flint::constant(1.123f, 20, 10, 2);
    Tensor<double, 3> t2 = Flint::constant(0.123, 20, 10, 2);
    Tensor<double, 3> t3 = ((t1 - t2) * M_PI).sin();
    for (int i = 0; i < 20; i++)
      for (int j = 0; j < 10; j++)
        for (int k = 0; k < 2; k++)
          CHECK_EQ(doctest::Approx(0.0), t3[i][j][k]);
    Tensor<double, 3> t4 = Flint::constant(1.0, 4, 2, 2);
    Tensor<double, 2> t5 = (t1 - t2).convolve(t4, 4, 2);
    for (int i = 0; i < t5.get_shape()[0]; i++)
      for (int j = 0; j < t5.get_shape()[1]; j++)
        CHECK_EQ(doctest::Approx(16.0), t5[i][j]);
    Tensor<double, 2> t6 = Flint::constant(1.0, 2, 4);
    Tensor<double, 3> t7 = (t1 - t2).matmul(t6);
    for (int i = 0; i < 20; i++)
      for (int j = 0; j < 10; j++)
        for (int k = 0; k < 4; k++)
          CHECK_EQ(doctest::Approx(2.0), t7[i][j][k]);
  }
  TEST_CASE("Basic Functions and Classes") {
    Tensor<float, 3> t1({{{0}, {1}}, {{2}, {3}}});
    Tensor<long, 1> t2({3});
    using namespace std;
    vector<vector<vector<float>>> od_t1 = *t1;
    vector<long> od_t2 = *t2;
    CHECK_EQ(od_t1[1][0][0], 2);
    CHECK_EQ(od_t1[1][1][0], 3);
    CHECK_EQ(od_t2[0], 3);

    Tensor<float, 3> t3 = t1 + t2;
    t3 = t3 + 7;
    //   test
    vector<vector<vector<float>>> foo = *t3;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        CHECK_EQ(i * 2 + j + 3 + 7, foo[i][j][0]);

    t3 = t3.pow(3);

    vector<float> bar = *t3.flattened();
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        CHECK_EQ(pow(i * 2 + j + 3 + 7, 3), bar[i * 2 + j]);

    Tensor<float, 2> t4 = t1.flattened(1);
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++)
        CHECK_EQ(i * 2 + j, t4[i * 2 + j][0]);
    }

    Tensor<double, 3> t5({{{0, 1, 2}, {1, 2, 3}}, {{2, 3, 4}, {3, 4, 5}}});
    Tensor<float, 2> t6({{0, 1}, {2, 3}, {4, 5}});
    Tensor<double, 3> t7 = t5.matmul(t6);

    vector<vector<vector<double>>> exp3{{{10, 13}, {16, 22}},
                                        {{22, 31}, {28, 40}}};
    vector<vector<vector<double>>> res3 = *t7;
    CHECK_EQ(exp3, res3);

    Tensor<double, 3> t8 = t6.matmul(t5);
    vector<vector<vector<double>>> exp4{
        {{1, 2, 3}, {3, 8, 13}, {5, 14, 23}},
        {{3, 4, 5}, {13, 18, 23}, {23, 32, 41}}};
    vector<vector<vector<double>>> res4 = *t8;
    CHECK_EQ(exp4, res4);
    CHECK_EQ(exp4.size(), res4.size());
    for (int i = 0; i < 2; i++) {
      CHECK_EQ(exp4[i].size(), res4[i].size());
      for (int j = 0; j < 3; j++) {
        CHECK_EQ(exp4[i][j].size(), res4[i][j].size());
        for (int k = 0; k < 3; k++) {
          CHECK_EQ(exp4[i][j][k], res4[i][j][k]);
        }
      }
    }
    Tensor<int, 7> large{{{{{{{0, 1}}}}}}};
    std::vector<std::vector<
        std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>>>
        res5 = *large;
    CHECK_EQ(1, large[0][0][0][0][0][0][1]);
    CHECK_EQ(1, res5[0][0][0][0][0][0][1]);
  }
  TEST_CASE("Parameter Communitivity") {
    using namespace std;
    Tensor<int, 3> t1{{{7, 1}, {1, 2}, {2, 3}}, {{1, 2}, {2, 3}, {3, 4}}};
    Tensor<int, 2> t2{{2, 9}, {3, 5}, {4, 3}};
    Tensor<int, 3> t3 = t1 + t2;
    Tensor<int, 3> t4 = t2 + t1;
    vector<vector<vector<int>>> r3 = *t3;
    vector<vector<vector<int>>> r4 = *t4;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 2; k++)
          CHECK_EQ(r3[i][j][k], r4[i][j][k]);
    // subtraction
    t3 = (-t1) + t2;
    t4 = t2 - t1;
    r3 = *t3;
    r4 = *t4;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 2; k++)
          CHECK_EQ(r3[i][j][k], r4[i][j][k]);
    // multiplication
    t3 = t1 * t2;
    t4 = t2 * t1;
    r4 = *t4;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 2; k++)
          CHECK_EQ(t3[i][j][k], r4[i][j][k]);
    // division with convert and indexing
    Tensor<double, 3> t5 = t1.convert<double>().pow(-1) * t2;
    Tensor<double, 3> t6 = t2 / t1.convert<double>();
    vector<vector<vector<double>>> r6 = *t6;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 2; k++)
          CHECK_EQ(t5[i][j][k],
                   doctest::Approx(r6[i][j][k]).epsilon(.000000000001));
    // power where exponent is higher dimensional
    Tensor<double, 1> t7{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    Tensor<long, 2> t8{{3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
                       {2, 2, 2, 2, 2, 2, 2, 2, 2, 2}};
    Tensor<double, 2> t9 = t7.pow(t8);
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 10; j++)
        CHECK_EQ(t9[i][j],
                 doctest::Approx(std::pow(j, 3 - i)).epsilon(.000000000001));
  }
  TEST_CASE("MIN, MAX") {
    Tensor<double, 2> t1{{1, 5}, {-3, 7}, {2, 3}};
    Tensor<double, 2> t2{{3, 1}, {2, -5}, {7, -9}};
    Tensor<double, 2> t3 = t1.min(t2);
    Tensor<double, 2> t4 = t2.max(t1);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 2; j++) {
        CHECK_EQ(t1[i][j] < t2[i][j] ? t1[i][j] : t2[i][j], t3[i][j]);
        CHECK_EQ(t1[i][j] > t2[i][j] ? t1[i][j] : t2[i][j], t4[i][j]);
      }
  }
  TEST_CASE("REPEAT") {
    Tensor<int, 3> t1{{{0}, {1}}, {{2}, {3}}};
    t1 = t1.repeat(1, 2);
    CHECK_EQ(4, t1.get_shape()[0]);
    CHECK_EQ(6, t1.get_shape()[1]);
    CHECK_EQ(1, t1.get_shape()[2]);
    using namespace std;
    vector<vector<vector<int>>> res = *t1;
    CHECK_EQ(0, res[0][0][0]);
    CHECK_EQ(1, res[0][1][0]);
    CHECK_EQ(0, res[0][2][0]);
    CHECK_EQ(1, res[0][5][0]);
    CHECK_EQ(2, res[3][4][0]);
    Tensor<int, 2> t2{{0, 1}, {2, 3}};
    t2 = t2.repeat(2, 3);
    CHECK_EQ(3, t2[1][5]);
    CHECK_EQ(0, t2[2][2]);
  }
  TEST_CASE("TRANSPOSE") {
    Tensor<int, 2> t1({{0, 1}, {2, 3}});
    t1 = t1.transpose();
    CHECK_EQ(0, t1[0][0]);
    CHECK_EQ(2, t1[0][1]);
    CHECK_EQ(1, t1[1][0]);
    CHECK_EQ(3, t1[1][1]);
    Tensor<double, 3> t2{{{1, 7}, {8, 8}, {2, 1}}, {{9, 3}, {2, 1}, {8, 9}}};
    t2 = t2.transpose({2, 1, 0});
    CHECK_EQ(9, t2[0][0][1]);
    CHECK_EQ(3, t2[1][0][1]);
    CHECK_EQ(8, t2[1][1][0]);
  }
  TEST_CASE("REDUCE OPERATIONS") {
    Tensor<double, 3> t1{{{1, 7}, {8, 8}, {2, 1}}, {{9, 3}, {2, 1}, {8, 9}}};
    Tensor<double, 2> t2 = t1.reduce_sum(0);
    CHECK_EQ(10, t2[0][0]);
    CHECK_EQ(10, t2[0][1]);
    CHECK_EQ(10, t2[2][1]);
    CHECK_EQ(10, t2[1][0]);
    CHECK_EQ(9, t2[1][1]);
    t2 = t1.reduce_sum(1);
    CHECK_EQ(11, t2[0][0]);
    CHECK_EQ(16, t2[0][1]);
    CHECK_EQ(19, t2[1][0]);
    CHECK_EQ(13, t2[1][1]);
    t2 = t1.reduce_sum(2);
    CHECK_EQ(8, t2[0][0]);
    CHECK_EQ(16, t2[0][1]);
    CHECK_EQ(12, t2[1][0]);
    CHECK_EQ(17, t2[1][2]);
    t2 = t1.reduce_mul(0);
    CHECK_EQ(9, t2[0][0]);
    CHECK_EQ(8, t2[1][1]);
    CHECK_EQ(16, t2[2][0]);
    t2 = t1.reduce_mul(1);
    CHECK_EQ(16, t2[0][0]);
    CHECK_EQ(27, t2[1][1]);
    CHECK_EQ(56, t2[0][1]);
    t2 = t1.reduce_mul(2);
    CHECK_EQ(7, t2[0][0]);
    CHECK_EQ(64, t2[0][1]);
    CHECK_EQ(2, t2[1][1]);
  }
  TEST_CASE("SLICE") {
    Tensor<long, 3> t1{{{1, 7}, {8, 8}, {2, 1}}, {{9, 3}, {2, 1}, {8, 9}}};
    Tensor<long, 3> s1 =
        t1.slice(TensorRange(), TensorRange(0, TensorRange::MAX_SCOPE, 2),
                 TensorRange(1, 2));
    Tensor<long, 3> s2 = s1.slice(TensorRange(0, 1));
    CHECK_EQ(1, s2.get_shape()[0]);
    CHECK_EQ(2, s2.get_shape()[1]);
    CHECK_EQ(1, s2.get_shape()[2]);
    CHECK_EQ(7, s2[0][0][0]);
    CHECK_EQ(1, s2[0][1][0]);
    // check flat data
    Tensor<double, 3> t2{{{-0.1}, {0.0}},
                         {{0.1}, {0.2}},
                         {{0.3}, {0.4}},
                         {{0.5}, {0.6}},
                         {{0.7}, {0.8}}};
    Tensor<double, 2> f1 = t2.flattened(2);
    // slice only positive to 0.6
    Tensor<double, 2> s3 = f1.slice(TensorRange(1, 4));
    Tensor<int, 1> f2 = (s3.flattened() * 10)
                            .slice(1, TensorRange::MAX_SCOPE, 2)
                            .convert<int>();
    CHECK_EQ(2, f2[0]);
    CHECK_EQ(4, f2[1]);
    CHECK_EQ(6, f2[2]);
    CHECK_EQ(3, f2.get_shape()[0]);
    // with negative indices
    Tensor<long, 3> i1 =
        t1.slice(TensorRange(-1, -3, -1), TensorRange(-1, -4, -2));
    Tensor<long, 1> t3 = i1.flattened().slice(-2, 0, -3);
    CHECK_EQ(2, t3.get_shape()[0]);
    CHECK_EQ(1, t3[0]);
    CHECK_EQ(3, t3[1]);
  }
  TEST_CASE("SQRT, EXP") {
    Tensor<long, 1> t1 = Tensor<long, 1>({12 * 12, 42 * 42, 420000l * 420000l})
                             .sqrt()
                             .convert<long>();
    CHECK_EQ(t1[0], 12);
    CHECK_EQ(t1[1], 42);
    CHECK_EQ(t1[2], 420000);
    Tensor<float, 4> t2 = Tensor<float, 4>{
        {{{0}, {1}}, {{2}, {3}}},
        {{{4}, {5}},
         {{6}, {7}}}}.sqrt();
    using doctest::Approx;
    CHECK_EQ(t2[0][0][0][0], 0);
    CHECK_EQ(t2[0][0][1][0], 1);
    CHECK_EQ(Approx(t2[0][1][0][0]).epsilon(0.00001), 1.41421);
    CHECK_EQ(Approx(t2[0][1][1][0]).epsilon(0.00001), 1.73205);
    CHECK_EQ(t2[1][0][0][0], 2);
    CHECK_EQ(Approx(t2[1][0][1][0]).epsilon(0.00001), 2.23607);
    CHECK_EQ(Approx(t2[1][1][0][0]).epsilon(0.00001), 2.44949);
    CHECK_EQ(Approx(t2[1][1][1][0]).epsilon(0.00001), 2.64575);
    Tensor<int, 2> t3{{0, 1}, {2, -1}};
    Tensor<double, 2> e3 = t3.exp();
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) {
        CHECK_EQ(Approx(exp(t3[i][j])), e3[i][j]);
      }
    Tensor<float, 2> t4{{0, 1}, {2, -1}};
    Tensor<float, 2> e4 = t4.exp();
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) {
        CHECK_EQ(Approx(exp(t4[i][j])), e4[i][j]);
      }
  }
  TEST_CASE("FABS") {
    Tensor<int, 2> t1{{-1, 3}, {-7, 9}};
    Tensor<int, 1> t2 = t1.abs().flattened();
    CHECK_EQ(1, t2[0]);
    CHECK_EQ(3, t2[1]);
    CHECK_EQ(7, t2[2]);
    CHECK_EQ(9, t2[3]);
  }
  TEST_CASE("FSIGN, FEVEN") {
    Tensor<long, 5> t1{{{{{-1, -3}, {4, 3}}}, {{{4, -2}, {-999, 0}}}}};
    Tensor<int, 5> s1 = t1.sign();
    CHECK_EQ(-1, s1[0][0][0][0][0]);
    CHECK_EQ(-1, s1[0][0][0][0][1]);
    CHECK_EQ(1, s1[0][0][0][1][0]);
    CHECK_EQ(1, s1[0][0][0][1][1]);
    CHECK_EQ(1, s1[0][1][0][0][0]);
    CHECK_EQ(-1, s1[0][1][0][0][1]);
    CHECK_EQ(-1, s1[0][1][0][1][0]);
    CHECK_EQ(1, s1[0][1][0][1][1]);
    Tensor<int, 5> e1 = t1.even();
    CHECK_EQ(0, e1[0][0][0][0][0]);
    CHECK_EQ(0, e1[0][0][0][0][1]);
    CHECK_EQ(1, e1[0][0][0][1][0]);
    CHECK_EQ(0, e1[0][0][0][1][1]);
    CHECK_EQ(1, e1[0][1][0][0][0]);
    CHECK_EQ(1, e1[0][1][0][0][1]);
    CHECK_EQ(0, e1[0][1][0][1][0]);
    CHECK_EQ(1, e1[0][1][0][1][1]);
    Tensor<long, 1> t2{-1, 2, 5, -8};
    Tensor<int, 1> s2 = t2.sign();
    CHECK_EQ(s2[0], -1);
    CHECK_EQ(s2[1], 1);
    CHECK_EQ(s2[2], 1);
    CHECK_EQ(s2[3], -1);
    Tensor<int, 1> e2 = t2.even();
    CHECK_EQ(e2[0], 0);
    CHECK_EQ(e2[1], 1);
    CHECK_EQ(e2[2], 0);
    CHECK_EQ(e2[3], 1);
    Tensor<float, 2> t3{{0.1}, {-9999.999}, {49.12345}, {-3.141592}};
    Tensor<int, 2> s3 = t3.sign();
    CHECK_EQ(1, s3[0][0]);
    CHECK_EQ(-1, s3[1][0]);
    CHECK_EQ(1, s3[2][0]);
    CHECK_EQ(-1, s3[3][0]);
  }
  TEST_CASE("FLESS, FGREATER, FEQUAL") {
    Tensor<int, 2> t1{{-1, 3, 1, -6}, {-7, 9, 5, -8}};
    Tensor<long, 1> t2{-1, 2, 5, -8};
    Tensor<int, 2> l12 = t1 < t2;
    CHECK_EQ(0, l12[0][0]);
    CHECK_EQ(0, l12[0][1]);
    CHECK_EQ(1, l12[0][2]);
    CHECK_EQ(0, l12[0][3]);
    CHECK_EQ(1, l12[1][0]);
    CHECK_EQ(0, l12[1][1]);
    CHECK_EQ(0, l12[1][2]);
    CHECK_EQ(0, l12[1][3]);
    Tensor<int, 2> g12 = t1 > t2;
    CHECK_EQ(0, g12[0][0]);
    CHECK_EQ(1, g12[0][1]);
    CHECK_EQ(0, g12[0][2]);
    CHECK_EQ(1, g12[0][3]);
    CHECK_EQ(0, g12[1][0]);
    CHECK_EQ(1, g12[1][1]);
    CHECK_EQ(0, g12[1][2]);
    CHECK_EQ(0, g12[1][3]);
    Tensor<int, 2> e12 = t1.equal(t2);
    CHECK_EQ(1, e12[0][0]);
    CHECK_EQ(0, e12[0][1]);
    CHECK_EQ(0, e12[0][2]);
    CHECK_EQ(0, e12[0][3]);
    CHECK_EQ(0, e12[1][0]);
    CHECK_EQ(0, e12[1][1]);
    CHECK_EQ(1, e12[1][2]);
    CHECK_EQ(1, e12[1][3]);
  }
  TEST_CASE("sin, cos, tan") {
    Tensor<int, 1> t1 = {0, 1, 2, 3};
    Tensor<double, 1> s1 = t1.convert<double>().sin();
    CHECK_EQ(doctest::Approx(0.0).epsilon(0.00001f), s1[0]);
    CHECK_EQ(doctest::Approx(0.8414709848078965).epsilon(0.00001), s1[1]);
    CHECK_EQ(doctest::Approx(0.9092974268256817).epsilon(0.00001), s1[2]);
    CHECK_EQ(doctest::Approx(0.1411200080598672).epsilon(0.00001), s1[3]);
    Tensor<double, 1> c1 = t1.convert<double>().cos();
    CHECK_EQ(doctest::Approx(1.000000).epsilon(0.00001), c1[0]);
    CHECK_EQ(doctest::Approx(0.540302).epsilon(0.00001), c1[1]);
    CHECK_EQ(doctest::Approx(-0.416147).epsilon(0.00001), c1[2]);
    CHECK_EQ(doctest::Approx(-0.989992).epsilon(0.00001), c1[3]);
    Tensor<double, 1> tan1 = t1.convert<double>().tan();
    CHECK_EQ(doctest::Approx(0.000000).epsilon(0.00001), tan1[0]);
    CHECK_EQ(doctest::Approx(1.557408).epsilon(0.00001), tan1[1]);
    CHECK_EQ(doctest::Approx(-2.185040).epsilon(0.00001), tan1[2]);
    CHECK_EQ(doctest::Approx(-0.142547).epsilon(0.00001), tan1[3]);
    Tensor<float, 2> t2 = {{0.2, 0.6, 0.3}, {0, .7, 1.0}};
    Tensor<float, 2> s2 = t2.sin().asin();
    Tensor<float, 2> c2 = t2.cos().acos();
    Tensor<float, 2> tan2 = t2.tan().atan();
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        CHECK_EQ(doctest::Approx(t2[i][j]).epsilon(0.00001f), s2[i][j]);
        CHECK_EQ(doctest::Approx(t2[i][j]).epsilon(0.00001f), c2[i][j]);
        CHECK_EQ(doctest::Approx(t2[i][j]).epsilon(0.00001f), tan2[i][j]);
      }
    }
  }
  TEST_CASE("FEXTEND") {
    Tensor<float, 2> a{{1, 2}, {3, 4}};
    a = a.extend(std::array<size_t, 2>{4, 4}, std::array<size_t, 2>{1, 2});
    for (int i = 0; i < 4; i++) {
      CHECK_EQ(0.0, a[0][i]);
      CHECK_EQ(0.0, a[3][i]);
    }
    CHECK_EQ(0, a[1][0]);
    CHECK_EQ(0, a[1][1]);
    CHECK_EQ(1, a[1][2]);
    CHECK_EQ(2, a[1][3]);
    CHECK_EQ(0, a[2][0]);
    CHECK_EQ(0, a[2][1]);
    CHECK_EQ(3, a[2][2]);
    CHECK_EQ(4, a[2][3]);
    Tensor<float, 2> b{{1, 2}, {3, 4}};
    b = b.extend(std::array<size_t, 2>{4, 7}, std::array<size_t, 2>{2, 1},
                 std::array<long, 2>{1, 2});
    for (int i = 0; i < 7; i++) {
      CHECK_EQ(0.0, b[0][i]);
      CHECK_EQ(0.0, b[1][i]);
    }
    CHECK_EQ(0.0, b[2][0]);
    CHECK_EQ(1.0, b[2][1]);
    CHECK_EQ(0.0, b[2][2]);
    CHECK_EQ(2.0, b[2][3]);
    CHECK_EQ(0.0, b[2][4]);
    CHECK_EQ(0.0, b[2][5]);
    CHECK_EQ(0.0, b[2][6]);
    CHECK_EQ(0.0, b[3][0]);
    CHECK_EQ(3.0, b[3][1]);
    CHECK_EQ(0.0, b[3][2]);
    CHECK_EQ(4.0, b[3][3]);
    CHECK_EQ(0.0, b[3][4]);
    CHECK_EQ(0.0, b[3][5]);
    CHECK_EQ(0.0, b[3][6]);
    Tensor<float, 2> c{{1, 2}, {3, 4}};
    c = c.extend(std::array<size_t, 2>{4, 7}, std::array<size_t, 2>{2, 1},
                 std::array<long, 2>{-1, -2});
    for (int i = 0; i < 7; i++) {
      CHECK_EQ(0.0, c[0][i]);
      CHECK_EQ(0.0, c[1][i]);
    }
    CHECK_EQ(0.0, c[2][0]);
    CHECK_EQ(4.0, c[2][1]);
    CHECK_EQ(0.0, c[2][2]);
    CHECK_EQ(3.0, c[2][3]);
    CHECK_EQ(0.0, c[2][4]);
    CHECK_EQ(0.0, c[2][5]);
    CHECK_EQ(0.0, c[2][6]);
    CHECK_EQ(0.0, c[3][0]);
    CHECK_EQ(2.0, c[3][1]);
    CHECK_EQ(0.0, c[3][2]);
    CHECK_EQ(1.0, c[3][3]);
    CHECK_EQ(0.0, c[3][4]);
    CHECK_EQ(0.0, c[3][5]);
    CHECK_EQ(0.0, c[3][6]);
  }
  TEST_CASE("REPEAT, REDUCE") {
    Tensor<int, 2> a{{1, 2}, {3, 4}};
    Tensor<int, 1> b = a.repeat(2, 2).reduce_mul(1); //{6, 14, 6, 14}
    CHECK_EQ(8, b[0]);
    CHECK_EQ(1728, b[1]);
    CHECK_EQ(8, b[2]);
    CHECK_EQ(1728, b[3]);

    b = a.reduce_mul(1).repeat(2); //{2, 12, 2, 12}
    CHECK_EQ(2, b[0]);
    CHECK_EQ(12, b[1]);
    CHECK_EQ(2, b[2]);
    CHECK_EQ(12, b[3]);
    // this crashes
    Tensor<int, 2> c = {{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}};
    Tensor<int, 2> d = c + a.reduce_mul(1).repeat(1);
    for (int i = 0; i < 4; i++) {
      CHECK_EQ(2, d[i][0]);
      CHECK_EQ(13, d[i][1]);
      CHECK_EQ(4, d[i][2]);
      CHECK_EQ(15, d[i][3]);
    }

    Tensor<int, 3> e{{{0, 1, 32}, {2, 3, 4}}, {{4, 5, -6}, {6, 7, -1}}};
    Tensor<int, 2> max1 = e.reduce_max(0)();
    CHECK_EQ(4, max1[0][0]);
    CHECK_EQ(5, max1[0][1]);
    CHECK_EQ(32, max1[0][2]);
    CHECK_EQ(6, max1[1][0]);
    CHECK_EQ(7, max1[1][1]);
    CHECK_EQ(4, max1[1][2]);
    Tensor<int, 2> max2 = e.reduce_max(1)();
    CHECK_EQ(2, max2[0][0]);
    CHECK_EQ(3, max2[0][1]);
    CHECK_EQ(32, max2[0][2]);
    CHECK_EQ(6, max2[1][0]);
    CHECK_EQ(7, max2[1][1]);
    CHECK_EQ(-1, max2[1][2]);
    Tensor<int, 2> max3 = e.reduce_max(2)();
    CHECK_EQ(32, max3[0][0]);
    CHECK_EQ(4, max3[0][1]);
    CHECK_EQ(5, max3[1][0]);
    CHECK_EQ(7, max3[1][1]);
    Tensor<int, 2> min1 = e.reduce_min(0)();
    CHECK_EQ(0, min1[0][0]);
    CHECK_EQ(1, min1[0][1]);
    CHECK_EQ(-6, min1[0][2]);
    CHECK_EQ(2, min1[1][0]);
    CHECK_EQ(3, min1[1][1]);
    CHECK_EQ(-1, min1[1][2]);
    Tensor<int, 2> min2 = e.reduce_min(1)();
    CHECK_EQ(0, min2[0][0]);
    CHECK_EQ(1, min2[0][1]);
    CHECK_EQ(4, min2[0][2]);
    CHECK_EQ(4, min2[1][0]);
    CHECK_EQ(5, min2[1][1]);
    CHECK_EQ(-6, min2[1][2]);
    Tensor<int, 2> min3 = e.reduce_min(2)();
    CHECK_EQ(0, min3[0][0]);
    CHECK_EQ(2, min3[0][1]);
    CHECK_EQ(-6, min3[1][0]);
    CHECK_EQ(-1, min3[1][1]);
    Tensor<int, 1> f{0, 1, 32, 2, 3, 4, -6, 7, -4};
    CHECK_EQ(-6, f.reduce_min()[0]);
    CHECK_EQ(32, f.reduce_max()[0]);
  }
}
TEST_CASE("Convolve") {
  Tensor<float, 3> t1{{{0, 1}, {1, 2}, {3, 4}},
                      {{5, 6}, {7, 8}, {9, 0}},
                      {{1, 2}, {3, 4}, {5, 6}}};
  Tensor<float, 3> k1{{{1, 1}, {2, 2}}, {{2, 2}, {1, 1}}};
  Tensor<float, 2> r1 = t1.convolve(k1, 1, 1);
  CHECK_EQ(44, r1[0][0]);
  CHECK_EQ(56, r1[0][1]);
  CHECK_EQ(25, r1[0][2]);
  CHECK_EQ(54, r1[1][0]);
  CHECK_EQ(58, r1[1][1]);
  CHECK_EQ(31, r1[1][2]);
  CHECK_EQ(17, r1[2][0]);
  CHECK_EQ(29, r1[2][1]);
  CHECK_EQ(11, r1[2][2]);
  Tensor<float, 3> t2{{{0}, {1}, {2}, {3}}, {{3}, {2}, {1}, {0}}};
  Tensor<float, 3> k2{{{1}, {2}}};
  Tensor<float, 2> r2 = t2.convolve(k2, 1, 2);
  CHECK_EQ(2, r2[0][0]);
  CHECK_EQ(8, r2[0][1]);
  CHECK_EQ(7, r2[1][0]);
  CHECK_EQ(1, r2[1][1]);
  // in context
  Tensor<float, 3> t4{{{0}, {1}}};
  Tensor<double, 3> k4{{{1}, {0}, {1}, {0}}};
  Tensor<double, 2> r4 =
      (t4 + 1).repeat(1, 1, 1).convolve(k4.pow(2).repeat(0, 0, 1));
  CHECK_EQ(4, r4[0][0]);
  CHECK_EQ(8, r4[0][1]);
  CHECK_EQ(2, r4[0][2]);
  CHECK_EQ(4, r4[0][3]);
  CHECK_EQ(4, r4[1][0]);
  CHECK_EQ(8, r4[1][1]);
  CHECK_EQ(2, r4[1][2]);
  CHECK_EQ(4, r4[1][3]);
}
TEST_CASE("Slide") {
  Tensor<float, 3> t1{{{0, 1}, {1, 2}, {3, 4}},
                      {{5, 6}, {7, 8}, {9, 0}},
                      {{1, 2}, {3, 4}, {5, 6}}};
  Tensor<float, 3> k1{{{1, 1}, {2, 2}}};
  Tensor<float, 3> r1 = t1.slide(k1);
  CHECK_EQ(34, r1[0][0][0]);
  CHECK_EQ(33, r1[0][0][1]);
  CHECK_EQ(56, r1[0][1][0]);
  CHECK_EQ(48, r1[0][1][1]);
  Tensor<float, 3> t2{{{0}, {1}, {2}, {3}, {4}}, {{4}, {3}, {2}, {1}, {0}}};
  Tensor<float, 3> k2{{{1}, {2}}};
  Tensor<float, 3> r2 = t2.slide(k2, 1, 2);
  CHECK_EQ(12, r2[0][0][0]);
  CHECK_EQ(16, r2[0][1][0]);
  Tensor<float, 3> k3{{{1, 1}}, {{2, 2}}};
  Tensor<float, 3> r3 = t1.slide(k3);
  CHECK_EQ(34, r3[0][0][0]);
  CHECK_EQ(33, r3[0][0][1]);
  CHECK_EQ(60, r3[0][1][0]);
  CHECK_EQ(52, r3[0][1][1]);
  // in context
  Tensor<float, 3> t4{{{0}, {1}}};
  Tensor<double, 3> k4{{{1}, {0}, {1}, {0}}};
  Tensor<double, 2> r4 =
      ((t4 + 1).repeat(1, 1, 1).slide(k4.pow(2).repeat(0, 0, 1)) + 1)
          .reduce_sum(2);
  CHECK_EQ(26, r4[0][0]);
  CHECK_EQ(2, r4[0][1]);
  CHECK_EQ(14, r4[0][2]);
  CHECK_EQ(2, r4[0][3]);
}
TEST_CASE("Total Reduce") {
  Tensor<float, 2> t1{{-1., 1.}, {1., 2.}, {4, 1}, {-0.5, -0.5}};
  Tensor<float, 1> r1 = t1.flattened().reduce_sum();
  CHECK_EQ(r1[0], 7);
  r1 = t1.flattened().reduce_mul();
  CHECK_EQ(r1[0], -2);
}
TEST_CASE("Concat") {
  Tensor<float, 2> t1{{-1., 1.}, {1., 2.}, {4, 1}, {-0.5, -0.5}};
  Tensor<float, 2> t2{{0, 0}, {3.141592, 42}};
  Tensor<float, 2> c1 = Flint::concat(t1, t2, 0);
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 2; j++)
      if (i < 4)
        CHECK_EQ(t1[i][j], c1[i][j]);
      else
        CHECK_EQ(t2[i - 4][j], c1[i][j]);
  }
  Tensor<float, 2> t3{{1, 2, 3, 4}, {5, 6, 7, 8}};
  Tensor<float, 2> c2 = Flint::concat(t2, t3, 1);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 6; j++) {
      if (j < 2)
        CHECK_EQ(t2[i][j], c2[i][j]);
      else
        CHECK_EQ(t3[i][j - 2], c2[i][j]);
    }
  }
  Tensor<float, 1> t7 = t2.slice(TensorRange(1)).flattened().repeat(1);
  Tensor<double, 2> t4 = t3.convert<double>() + t7;
  Tensor<double, 2> t5 = t4 - t3;
  Tensor<double, 2> t6 = Flint::concat(t4, t5, 0);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      CHECK_EQ(doctest::Approx(t6[i][j]).epsilon(0.00001),
               (i >= 2) ? (j % 2 == 0 ? 3.141592 : 42)
                        : t3[i][j] + (j % 2 == 0 ? 3.141592 : 42));
    }
  }
}
TEST_CASE("Random") {
  Tensor<double, 4> r1 = Flint::random(4, 4, 4, 4) + 1.0;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 4; k++)
        for (int l = 0; l < 4; l++)
          CHECK_LE(1.0, r1[i][j][k][l]);
}
TEST_CASE("Saving and Loading to files") {
  Tensor<double, 3> a = Flint::constant(3.0, 9, 4, 1);
  Tensor<float, 2> b{{1}, {-1}, {2}, {-2}};
  Tensor<double, 3> c = a + b;
  std::ofstream ofile;
  ofile.open("test.flint");
  ofile << c;
  ofile.close();
  std::ifstream ifile;
  ifile.open("test.flint");
  Tensor<double, 3> e = Tensor<double, 3>::read_from(ifile);
  ifile.close();
  for (int i = 0; i < 9; i++)
    for (int j = 0; j < 4; j++)
      CHECK_EQ(e[i][j][0], c[i][j][0]);
  std::remove("test.flint");
}
TEST_CASE("Expand") {
  Tensor<double, 2> a = {{0, 1}, {2, 3}};
  Tensor<double, 3> e1 = a.expand(0, 3)();
  Tensor<double, 3> e2 = a.expand(1, 3)();
  Tensor<double, 3> e3 = a.expand(2, 3)();
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++) {
        CHECK_EQ(a[j][k], e1[i][j][k]);
        CHECK_EQ(a[j][k], e2[j][i][k]);
        CHECK_EQ(a[j][k], e3[j][k][i]);
      }
}
TEST_CASE("Index") {
  Tensor<double, 3> a = {
      {{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}, {{8, 9}, {10, 11}}};
  Tensor<int, 1> i1 = {0, 2};
  Tensor<double, 3> a1 = a.index(i1);
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++)
        CHECK_EQ(a1[i][j][k], i == 0 ? j * 2 + k : 8 + j * 2 + k);
  Tensor<int, 1> i2 = {0, 1, 1, 2};
  Tensor<double, 3> a2 = a.multi_index(i2);
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++)
        CHECK_EQ(a2[i][j][k],
                 i == 0 ? j * 2 + k : (i < 3 ? 4 + j * 2 + k : 8 + j * 2 + k));
  Tensor<int, 2> i3 = {{0}, {1}, {0}};
  Tensor<double, 3> a3 = a.index(i3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 1; j++)
      for (int k = 0; k < 2; k++)
        CHECK_EQ(a3[i][j][k], i == 0 ? k : i == 1 ? 6 + k : 8 + k);
  Tensor<int, 3> i4 = {{{0, 0}, {1, 0}}, {{0, 1}, {1, 1}}, {{1, 1}, {0, 0}}};
  Tensor<double, 3> a4 = a.multi_index(i4);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++)
        CHECK_EQ(a4[i][j][k], a[i][j][i4[i][j][k]]);
  Tensor<int, 2> i5 = {{0, 0, 1, 1}, {1, 0, 1, 0}, {0, 1, 1, 0}};
  Tensor<double, 3> a5 = a.multi_index(i5);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 2; k++)
        CHECK_EQ(a5[i][j][k], a[i][i5[i][j]][k]);
  Tensor<double, 3> c = {{{1, 2, 3}}};
  c = c.repeat(2, 5, 1); // new shape: [3, 6, 6]
  c = c + 2;
  c = c.max(4);
  Tensor<int, 2> i6 = {{4, 5}, {3, 3}, {0, 1}};
  Tensor<double, 3> c1 = c.multi_index(i6);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 6; k++)
        CHECK_EQ(c1[i][j][k], c[i][i6[i][j]][k]);
}
TEST_CASE("Test Example 1") {
  Tensor<float, 2> t1{{-1., 0.}, {1., 2.}};
  Tensor<float, 1> c1{4.0f, 4.0f};
  t1 = t1 + c1 - 1.0f;
  Tensor<double, 3> t2{{{0.0, 1.0}, {2.0, 3.0}}, {{4.0, 5.0}, {6.0, 7.0}}};
  Tensor<double, 3> t3 = t2.matmul(t1);
  Tensor<double, 3> t4 =
      t3.slice(TensorRange(0, 1), TensorRange(0, 1), TensorRange(0, 2));
  Tensor<double, 1> t5 = t4.reduce_mul(2).flattened();
  CHECK_EQ(20, t5[0]);
}
TEST_SUITE("Index operations and broadcasting") {
  TEST_CASE("Slice") {
    Tensor<float, 3> t1{{{0, 1}, {1, 2}, {3, 4}},
                        {{5, 6}, {7, 8}, {9, 0}},
                        {{-1, -2}, {-3, -4}, {-5, -6}}};
    Tensor<float, 3> o =
        t1 * t1.slice(TensorRange(0, 1), TensorRange(0, 1), TensorRange(0, 2))
                 .flattened();
    auto exp = std::vector<std::vector<std::vector<float>>>{
        {{0.000000, 1.000000}, {0.000000, 2.000000}, {0.000000, 4.000000}},
        {{0.000000, 6.000000}, {0.000000, 8.000000}, {0.000000, 0.000000}},
        {{-0.000000, -2.000000},
         {-0.000000, -4.000000},
         {-0.000000, -6.000000}}};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 2; k++)
          CHECK_EQ(doctest::Approx(exp[i][j][k]), o[i][j][k]);
      }
    }
  }
  TEST_CASE("Repeat") {
    Tensor<float, 2> t1{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 0, 1}};
    Tensor<float, 1> t2{2, 7};
    Tensor<float, 2> o = t1 + t2.repeat(1);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 4; j++)
        CHECK_EQ(t1[i][j] + (j % 2 == 0 ? 2 : 7), o[i][j]);
  }
  TEST_CASE("Transpose") {
    Tensor<float, 2> t1{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 0, 1}};
    Tensor<float, 1> t2{2, 7, 8};
    Tensor<float, 2> o = t1.transpose() + (t2 - 1);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 4; j++)
        CHECK_EQ(t1[i][j] + t2[i] - 1, o[j][i]);

    Tensor<float, 3> t3{{{0, 1, 2}, {2, 3, 4}, {5, 6, 7}, {8, 9, -1}},
                        {{-3, -4, -5}, {-2, -6, -7}, {-8, -9, 0}, {1, 2, 3}}};
    Tensor<float, 3> t4 = (t3 * t1.transpose());
    for (int k = 0; k < 2; k++)
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
          CHECK_EQ(t3[k][i][j] * t1[j][i], t4[k][i][j]);
  }
}
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
  if (eager)
    fEnableEagerExecution();
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  int res;
  if (doCPU) {
    flintInit(FLINT_BACKEND_ONLY_CPU);
    res = context.run();
    flintCleanup();
  }
  if (doGPU) {
    flintInit(FLINT_BACKEND_ONLY_GPU);
    res = context.run();
    flintCleanup();
  }

  if (context.shouldExit())
    return res;
  int client_stuff_return_code = 0;

  return res + client_stuff_return_code;
}
