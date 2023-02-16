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
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "testutils.hpp"
#include <vector>

#include "../flint.hpp"
TEST_SUITE("Graph implementation") {
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
      REQUIRE(gn1->operation);
      CHECK_EQ(gn1->operation->data_type, F_FLOAT64);
      FGraphNode *right1 = gn1->predecessors[1];
      CHECK_EQ(right1->num_predecessor, 0);
      CHECK(right1->predecessors == nullptr);
      REQUIRE(right1->operation);
      FStore *store1 = (FStore *)right1->operation->additional_data;
      CHECK_EQ(right1->operation->data_type, F_FLOAT32);
      CHECK_EQ(store1->num_entries, 100);
      fFreeGraph(gn1);
    }
    {
      vector<long> v1(100);
      vector<int> v2(100);
      // construct graph 2
      vector<size_t> shape = {10, 10};
      FGraphNode *gn2 =
          fCreateGraph(v1.data(), v1.size(), F_INT64, shape.data(), 2);
      gn2 = fsub(gn2, 7.0);
      FGraphNode *gn21 =
          fCreateGraph(v2.data(), v2.size(), F_INT32, shape.data(), 2);
      gn2 = fdiv(gn2, gn21);
      // test
      REQUIRE_EQ(gn2->num_predecessor, 2);
      REQUIRE(gn2->operation);
      CHECK_EQ(gn2->operation->op_type, FDIV);
      CHECK_EQ(gn2->operation->data_type, F_INT64);
      FGraphNode *right2 = gn2->predecessors[1];
      CHECK_EQ(right2->num_predecessor, 0);
      CHECK(right2->predecessors == nullptr);
      REQUIRE(gn2->operation);
      FStore *store2 = (FStore *)right2->operation->additional_data;
      CHECK_EQ(right2->operation->data_type, F_INT32);
      CHECK_EQ(store2->num_entries, 100);
      FGraphNode *left1 = gn2->predecessors[0];
      FGraphNode *const1 = left1->predecessors[1];
      CHECK_EQ(const1->operation->op_type, FCONST);
      fFreeGraph(gn2);
    }
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
    FGraphNode *result = fExecuteGraph(gn1);
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
    result = fExecuteGraph(gn3);
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
    FGraphNode *result = fExecuteGraph(gn3);
    FResultData *rd = result->result_data;
    CHECK_EQ(rd->num_entries, 9);
    REQUIRE_EQ(result->operation->dimensions, 2);
    CHECK_EQ(result->operation->shape[0], 3);
    CHECK_EQ(result->operation->shape[1], 3);
    CHECK_EQ(result->operation->data_type, F_FLOAT64);
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
      newResult = fExecuteGraph(gn5);
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

    FGraphNode *r1 = fExecuteGraph(g2);
    FGraphNode *r3 = fExecuteGraph(g4);
    FGraphNode *r2 = fExecuteGraph(g3);
    CHECK_EQ(2, r3->operation->dimensions);
    CHECK_EQ(3, r3->operation->shape[0]);
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
    g = fExecuteGraph(g);
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
    g1 = fExecuteGraph(g1);
    g2 = fExecuteGraph(g2);
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
    g1 = fExecuteGraph(fneg(g1));
    g2 = fExecuteGraph(fneg(g2));
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
    FGraphNode *r1 = fExecuteGraph(mm1);
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
    REQUIRE_EQ(mm2->operation->shape[0], s3[0]);
    REQUIRE_EQ(mm2->operation->shape[1], s3[1]);
    FGraphNode *r2 = fExecuteGraph(mm2);
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
    REQUIRE_EQ(mm2->operation->shape[0], 2);
    REQUIRE_EQ(mm2->operation->shape[1], 2);
    REQUIRE_EQ(mm2->operation->shape[2], 2);
    r2 = fExecuteGraph(mm2);
    FResultData *rd3 = r2->result_data;
    double *d3 = (double *)rd3->data;
    for (size_t i = 0; i < rd3->num_entries; i++)
      CHECK_EQ(fe3[i], d3[i]);
    fFreeGraph(r2);
  }
}
#include <chrono>
TEST_SUITE("C++ Bindings") {
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
    CHECK_EQ((std::string)t3,
             "Tensor<FLOAT32, shape: [2, 2, 1]>(<not yet executed>)");
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
        t1.slice(TensorRange(), TensorRange(0, TensorRange::MAX_SIZE, 2),
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
    Tensor<int, 1> f2 =
        (s3.flattened() * 10).slice(1, TensorRange::MAX_SIZE, 2).convert<int>();
    CHECK_EQ(2, f2[0]);
    CHECK_EQ(4, f2[1]);
    CHECK_EQ(6, f2[2]);
    CHECK_EQ(3, f2.get_shape());
    // with negative indices
    Tensor<long, 1> t3 =
        t1.slice(TensorRange(-1, -3, -1), TensorRange(-1, -4, -2))
            .flattened()
            .slice(-2, 0, -3);
    CHECK_EQ(2, t3.get_shape());
    CHECK_EQ(1, t3[0]);
    CHECK_EQ(3, t3[1]);
  }
  TEST_CASE("FABS") {
    Tensor<int, 2> t1{{-1, 3}, {-7, 9}};
    Tensor<int, 1> t2 = t1.abs().flattened();
    CHECK_EQ(1, t2[0]);
    CHECK_EQ(3, t2[1]);
    CHECK_EQ(7, t2[2]);
    CHECK_EQ(9, t2[3]);
  }
}
TEST_CASE("Eager Execution") {
  enable_eager_execution();
  Tensor<float, 2> t1{{-1., 0.}, {1., 2.}};
  t1 = t1 + 3.0f;
  Tensor<double, 3> t2{{{0.0, 1.0}, {2.0, 3.0}}, {{4.0, 5.0}, {6.0, 7.0}}};
  Tensor<double, 3> t3 = t2.matmul(t1);
  Tensor<double, 3> t4 =
      t3.slice(TensorRange(0, 1), TensorRange(0, 1), TensorRange(0, 2));
  Tensor<double, 1> t5 = t4.reduce_mul(2).flattened();
  CHECK_EQ(20, t5[0]);
  disable_eager_execution();
}
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
  disable_eager_execution();
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

  if (context.shouldExit())
    return res;
  int client_stuff_return_code = 0;

  return res + client_stuff_return_code;
}
