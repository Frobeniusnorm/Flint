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

TEST_SUITE("Graph implementation") {
  TEST_CASE("createGraph, add, mul, sub, div") {
    using namespace std;
    {
      vector<double> v1(100);
      vector<float> v2(100);
      // construct graph 1
      vector<int> shape{100};
      FGraphNode *gn1 =
          createGraph(v1.data(), v1.size(), FLOAT64, shape.data(), 1);
      gn1 = add(gn1, 7.0);
      FGraphNode *gn12 =
          createGraph(v2.data(), v2.size(), FLOAT32, shape.data(), 1);
      gn1 = mul(gn1, gn12);
      freeGraph(gn12);
      // test
      REQUIRE_EQ(gn1->num_predecessor, 2);
      REQUIRE(gn1->operation);
      CHECK_EQ(gn1->operation->data_type, FLOAT64);
      FGraphNode *right1 = gn1->predecessors[1];
      CHECK_EQ(right1->num_predecessor, 0);
      CHECK(right1->predecessors == nullptr);
      REQUIRE(right1->operation);
      FStore *store1 = (FStore *)right1->operation->additional_data;
      CHECK_EQ(right1->operation->data_type, FLOAT32);
      CHECK_EQ(store1->num_entries, 100);
      freeGraph(gn1);
    }
    {
      vector<long> v1(100);
      vector<int> v2(100);
      // construct graph 2
      vector<int> shape = {10, 10};
      FGraphNode *gn2 =
          createGraph(v1.data(), v1.size(), INT64, shape.data(), 2);
      gn2 = sub(gn2, 7.0);
      FGraphNode *gn21 =
          createGraph(v2.data(), v2.size(), INT32, shape.data(), 2);
      gn2 = div(gn2, gn21);
      // test
      REQUIRE_EQ(gn2->num_predecessor, 2);
      REQUIRE(gn2->operation);
      CHECK_EQ(gn2->operation->op_type, DIV);
      CHECK_EQ(gn2->operation->data_type, INT64);
      FGraphNode *right2 = gn2->predecessors[1];
      CHECK_EQ(right2->num_predecessor, 0);
      CHECK(right2->predecessors == nullptr);
      REQUIRE(gn2->operation);
      FStore *store2 = (FStore *)right2->operation->additional_data;
      CHECK_EQ(right2->operation->data_type, INT32);
      CHECK_EQ(store2->num_entries, 100);
      FGraphNode *left1 = gn2->predecessors[0];
      FGraphNode *const1 = left1->predecessors[1];
      CHECK_EQ(const1->operation->op_type, CONST);
      freeGraph(gn2);
    }
  }
}
TEST_SUITE("Execution") {
  TEST_CASE("init, execution (add, sub, mul) and cleanup") {
    using namespace std;
    vector<double> v1(10, 4.0);
    vector<float> v2(10, 4.0f);
    // construct graph 1
    vector<int> shape{10};
    FGraphNode *gn1 =
        createGraph(v1.data(), v1.size(), FLOAT64, shape.data(), 1);
    gn1 = add(gn1, 7.0);
    FGraphNode *gn11 =
        createGraph(v2.data(), v2.size(), FLOAT32, shape.data(), 1);
    gn1 = mul(gn1, gn11);
    freeGraph(gn11); // delete handle
    FGraphNode *result = executeGraph(gn1);
    FResultData *rd = (FResultData *)result->operation->additional_data;
    CHECK_EQ(rd->num_entries, 10);
    for (int i = 0; i < rd->num_entries; i++)
      CHECK_EQ(((double *)rd->data)[i], 44);
    // construct graph 2 (first not-tree)
    vector<float> v3(10);
    for (int i = 0; i < 10; i++)
      v3[i] = i + 1;
    FGraphNode *gn2 =
        createGraph(v3.data(), v3.size(), FLOAT32, shape.data(), 1);
    FGraphNode *gn3 = add(gn2, result);
    gn3 = add(gn3, result);
    gn3 = sub(gn3, 80);
    gn3 = add(gn3, gn2);
    result = executeGraph(gn3);
    rd = (FResultData *)result->operation->additional_data;
    CHECK_EQ(rd->num_entries, 10);
    for (int i = 0; i < 10; i++)
      CHECK_EQ(((double *)rd->data)[i], 8 + (i + 1) * 2);
    freeGraph(result);
  }
  TEST_CASE("Multidimensional Data") {
    using namespace std;
    vector<vector<double>> v1{
        {0.0, 1.0, 2.0}, {0.0, -1.0, -2.0}, {0.0, 1.0, 2.0}};
    vector<vector<double>> v2{
        {2.0, 1.0, 0.0}, {0.0, -1.0, -2.0}, {2.0, 1.0, 2.0}};
    vector<double> f1 = flattened(v1);
    vector<double> f2 = flattened(v2);
    vector<int> shape{3, 3};
    FGraphNode *gn1 =
        createGraph(f1.data(), f1.size(), FLOAT64, shape.data(), 2);
    FGraphNode *gn2 =
        createGraph(f2.data(), f2.size(), FLOAT64, shape.data(), 2);
    FGraphNode *gn3 = add(gn1, gn2);
    FGraphNode *result = executeGraph(gn3);
    FResultData *rd = (FResultData *)result->operation->additional_data;
    CHECK_EQ(rd->num_entries, 9);
    REQUIRE_EQ(result->operation->dimensions, 2);
    CHECK_EQ(result->operation->shape[0], 3);
    CHECK_EQ(result->operation->shape[1], 3);
    CHECK_EQ(result->operation->data_type, FLOAT64);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        CHECK_EQ(((double *)rd->data)[i * 3 + j], v1[i][j] + v2[i][j]);
    // 3d + 2d
    vector<vector<vector<int>>> v3{{{0, 1, 2}, {2, 1, 0}, {0, 1, 2}},
                                   {{5, 9, 2}, {3, 5, 7}, {3, 4, 1}},
                                   {{0, 1, 2}, {9, 8, 7}, {5, 9, 7}},
                                   {{-3, -2, 4}, {-1, -2, 3}, {11, 1, 0}}};

    vector<int> f3 = flattened(v3);
    vector<int> shape_f3{4, 3, 3};
    FGraphNode *gn4 =
        createGraph(f3.data(), f3.size(), INT32, shape_f3.data(), 3);
    FGraphNode *gn5 = add(gn4, result);
    FGraphNode *newResult[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; i++) {
      newResult[i] = executeGraph(gn5);
      rd = (FResultData *)newResult[i]->operation->additional_data;

      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
          for (int k = 0; k < 3; k++)
            CHECK_EQ(((double *)rd->data)[i * 9 + j * 3 + k],
                     v1[j][k] + v2[j][k] + v3[i][j][k]);
    }
    freeGraph(newResult[0]);
    freeGraph(newResult[1]);
  }
  TEST_CASE("pow") {
    using namespace std;
    vector<int> s1{3, 2};
    int s2 = 2;
    vector<vector<int>> d1{{1, 3}, {0, 8}, {-3, -3}};
    vector<int> f1 = flattened(d1);
    vector<long> d2{2, 1};
    vector<vector<float>> d3{{0, 2}, {1, 0}, {-1, 2}};
    vector<float> f3 = flattened(d3);
    FGraphNode *g1 = createGraph(f1.data(), f1.size(), INT32, s1.data(), 2);
    FGraphNode *g12 = createGraph(d2.data(), d2.size(), INT64, &s2, 1);
    FGraphNode *g2 = pow(g1, g12);
    FGraphNode *g13 = createGraph(f3.data(), f3.size(), FLOAT32, s1.data(), 2);
    FGraphNode *g3 = pow(g1, g13);
    freeGraph(g13); // delete handles
    freeGraph(g12);
    FGraphNode *g4 = pow(g1, 2);
    vector<vector<long>> e1{{1, 3}, {0, 8}, {9, -3}};
    vector<vector<float>> e2{{1, 9}, {0, 1}, {-0.3333333333333333, 9}};
    vector<vector<int>> e3{{1, 9}, {0, 64}, {9, 9}};

    FGraphNode *r1 = executeGraph(g2);
    FGraphNode *r3 = executeGraph(g4);
    FGraphNode *r2 = executeGraph(g3);

    FResultData *res = (FResultData *)r1->operation->additional_data;
    long *ldata = (long *)res->data;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 2; j++)
        CHECK_EQ(ldata[i * 2 + j], e1[i][j]);

    res = (FResultData *)r2->operation->additional_data;
    float *fdata = (float *)res->data;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 2; j++)
        CHECK_EQ(fdata[i * 2 + j], e2[i][j]);

    res = (FResultData *)r3->operation->additional_data;
    int *idata = (int *)res->data;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 2; j++)
        CHECK_EQ(idata[i * 2 + j], e3[i][j]);
    freeGraph(r1);
    freeGraph(r2);
    freeGraph(r3);
  }
  TEST_CASE("flatten") {
    using namespace std;
    vector<vector<int>> d1{{1, 3}, {0, 8}, {-3, -3}};
    vector<int> f1 = flattened(d1);
    vector<int> d2{3, 3, 4, 4, 5, 5};
    vector<int> e1{4, 6, 4, 12, 2, 2};
    vector<int> s1{3, 2};
    int s2 = 6;
    FGraphNode *g = createGraph(f1.data(), f1.size(), INT32, s1.data(), 2);
    FGraphNode *gi = createGraph(d2.data(), d2.size(), INT32, &s2, 1);
    g = add(flatten(g), gi);
    freeGraph(gi);
    g = executeGraph(g);
    FResultData *res = (FResultData *)g->operation->additional_data;
    int *data = (int *)res->data;
    for (int i = 0; i < 6; i++)
      CHECK_EQ(data[i], e1[i]);
    freeGraph(g);
    // flatten with index
    vector<vector<vector<int>>> d3{{{0, 1}, {2, 3}, {4, 5}},
                                   {{6, 7}, {8, 9}, {10, 11}}};
    vector<int> f3 = flattened(d3);
    vector<int> s3{2, 3, 2};
    vector<vector<int>> d4{{3, 3, 4, 4, 5, 5}, {5, 5, 4, 4, 3, 3}};
    vector<int> f4 = flattened(d4);
    vector<int> s4{2, 6};
    vector<vector<int>> d5{{3, 3}, {4, 4}, {5, 5}, {5, 5}, {4, 4}, {3, 3}};
    vector<int> f5 = flattened(d5);
    vector<int> s5{6, 2};
    g = createGraph(f3.data(), f3.size(), INT32, s3.data(), 3);
    FGraphNode *g1 = flatten(g, 2);
    FGraphNode *g2 = flatten(g, 1);
    FGraphNode *g11 = createGraph(f4.data(), f4.size(), INT32, s4.data(), 2);
    FGraphNode *g21 = createGraph(f5.data(), f5.size(), INT32, s5.data(), 2);
    g1 = flatten(add(g1, g11));
    g2 = flatten(add(g2, g21));
    freeGraph(g11);
    freeGraph(g21);
    vector<int> exp{3, 4, 6, 7, 9, 10, 11, 12, 12, 13, 13, 14};
    g1 = executeGraph(g1);
    // g2 = executeGraph(g2);
    int *r1 = (int *)((FResultData *)g1->operation->additional_data)->data;
    // int *r2 = (int *)((FResultData *)g2->operation->additional_data)->data;
    for (int i = 0; i < 12; i++) {
      CHECK_EQ(r1[i], exp[i]);
      // CHECK_EQ(r2[i], exp[i]);
    }
    freeGraph(g1);
    freeGraph(g2);
  }
}
#include "../flint.hpp"
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

    foo = *t3;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        CHECK_EQ(pow(i * 2 + j + 3 + 7, 3), foo[i][j][0]);
  }
}
TEST_SUITE("Benchmarking") {
  TEST_CASE("Benchmark 1") {
    using namespace std;
    vector<vector<vector<double>>> data = vector<vector<vector<double>>>(100);
#pragma omp parallel for
    for (int i = 0; i < 100; i++) {
      data[i] = vector<vector<double>>(100);
      for (int j = 0; j < 100; j++) {
        data[i][j] = vector<double>(10);
        for (int k = 0; k < 10; k++)
          data[i][j][k] = i / 100.0 + 100.0 / j + 10.0 / k;
      }
    }
    vector<vector<long>> data2 = vector<vector<long>>(100);
    for (int i = 0; i < 100; i++) {
      data2[i] = vector<long>(10);
      for (int j = 0; j < 10; j++) {
        data2[i][j] = 12345678;
      }
    }
    for (int i = 0; i < 2; i++) {
      Tensor<double, 3> t1(data);
      t1 = t1 + 1234567.8;
      Tensor<long, 2> t2(data2);
      t2 = t2 - 2000000;
      t1 = t1 / t2;
      t1 = t1.pow(0.5);
      auto start = std::chrono::high_resolution_clock::now();
      if (i == 0)
        t1.execute_cpu();
      else
        t1.execute_gpu();
      chrono::duration<double, std::milli> elapsed =
          chrono::high_resolution_clock::now() - start;
      log(INFO, (i == 0 ? string("CPU") : string("GPU")) +
                    " implementation took " + to_string(elapsed.count()) +
                    "ms!");
    }
  }
}
int main(int argc, char **argv) {
  flintInit(1, 1);
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  int res = context.run();
  flintCleanup();
  if (context.shouldExit())
    return res;
  int client_stuff_return_code = 0;

  return res + client_stuff_return_code;
}
