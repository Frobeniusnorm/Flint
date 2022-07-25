#include "../flint.hpp"
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "testutils.hpp"
#include <vector>

TEST_SUITE("Graph implementation") {
  TEST_CASE("createGraph, add, mul, sub, div") {
    using namespace std;
    using namespace FlintBackend;
    {
      vector<double> v1(100);
      vector<float> v2(100);
      // construct graph 1
      vector<int> shape{100};
      GraphNode *gn1 =
          createGraph(v1.data(), v1.size(), FLOAT64, shape.data(), 1);
      gn1 = add(gn1, 7.0);
      gn1 =
          mul(gn1, createGraph(v2.data(), v2.size(), FLOAT32, shape.data(), 1));
      // test
      REQUIRE_EQ(gn1->num_predecessor, 2);
      REQUIRE(gn1->operation);
      CHECK_EQ(gn1->operation->data_type, FLOAT64);
      GraphNode *right1 = gn1->predecessors[1];
      CHECK_EQ(right1->num_predecessor, 0);
      CHECK(right1->predecessors == nullptr);
      REQUIRE(right1->operation);
      Store *store1 = (Store *)right1->operation;
      CHECK_EQ(store1->data_type, FLOAT32);
      CHECK_EQ(store1->num_entries, 100);
      CHECK_EQ(store1->data, v2.data());
      freeGraph(gn1);
    }
    {
      vector<long> v1(100);
      vector<int> v2(100);
      // construct graph 2
      vector<int> shape = {10, 10};
      GraphNode *gn2 =
          createGraph(v1.data(), v1.size(), INT64, shape.data(), 2);
      gn2 = sub(gn2, 7.0);
      gn2 = div(gn2, createGraph(v2.data(), v2.size(), INT32, shape.data(), 2));
      // test
      REQUIRE_EQ(gn2->num_predecessor, 2);
      REQUIRE(gn2->operation);
      CHECK_EQ(gn2->operation->op_type, DIV);
      CHECK_EQ(gn2->operation->data_type, INT64);
      GraphNode *right2 = gn2->predecessors[1];
      CHECK_EQ(right2->num_predecessor, 0);
      CHECK(right2->predecessors == nullptr);
      REQUIRE(gn2->operation);
      Store *store2 = (Store *)right2->operation;
      CHECK_EQ(store2->data_type, INT32);
      CHECK_EQ(store2->num_entries, 100);
      CHECK_EQ(store2->data, v2.data());
      GraphNode *left1 = gn2->predecessors[0];
      GraphNode *const1 = left1->predecessors[1];
      CHECK_EQ(const1->operation->op_type, CONST);
      freeGraph(gn2);
    }
  }
}
TEST_SUITE("Execution") {
  TEST_CASE("init, execution (add, sub, mul) and cleanup") {
    using namespace FlintBackend;
    using namespace std;
    vector<double> v1(10, 4.0);
    vector<float> v2(10, 4.0f);
    // construct graph 1
    vector<int> shape{10};
    GraphNode *gn1 =
        createGraph(v1.data(), v1.size(), FLOAT64, shape.data(), 1);
    gn1 = add(gn1, 7.0);
    gn1 = mul(gn1, createGraph(v2.data(), v2.size(), FLOAT32, shape.data(), 1));
    GraphNode *result = executeGraph(gn1);
    ResultData *rd = (ResultData *)result->operation;
    CHECK_EQ(rd->num_entries, 10);
    for (int i = 0; i < rd->num_entries; i++)
      CHECK_EQ(((double *)rd->data)[i], 44);
    // construct graph 2 (first not-tree)
    vector<float> v3(10);
    for (int i = 0; i < 10; i++)
      v3[i] = i + 1;
    GraphNode *gn2 =
        createGraph(v3.data(), v3.size(), FLOAT32, shape.data(), 1);
    GraphNode *gn3 = add(gn2, result);
    gn3 = add(gn3, result);
    gn3 = sub(gn3, 80);
    gn3 = add(gn3, gn2);
    result = executeGraph(gn3);
    rd = (ResultData *)result->operation;
    CHECK_EQ(rd->num_entries, 10);
    for (int i = 0; i < 10; i++)
      CHECK_EQ(((double *)rd->data)[i], 8 + (i + 1) * 2);
    freeGraph(result);
  }
  TEST_CASE("Multidimensional Data") {
    using namespace FlintBackend;
    using namespace std;
    vector<vector<double>> v1{
        {0.0, 1.0, 2.0}, {0.0, -1.0, -2.0}, {0.0, 1.0, 2.0}};
    vector<vector<double>> v2{
        {2.0, 1.0, 0.0}, {0.0, -1.0, -2.0}, {2.0, 1.0, 2.0}};
    vector<double> f1 = flattened(v1);
    vector<double> f2 = flattened(v2);
    vector<int> shape{3, 3};
    GraphNode *gn1 =
        createGraph(f1.data(), f1.size(), FLOAT64, shape.data(), 2);
    GraphNode *gn2 =
        createGraph(f2.data(), f2.size(), FLOAT64, shape.data(), 2);
    GraphNode *gn3 = add(gn1, gn2);
    GraphNode *result = executeGraph(gn3);
    ResultData *rd = (ResultData *)result->operation;
    CHECK_EQ(rd->num_entries, 9);
    REQUIRE_EQ(rd->dimensions, 2);
    CHECK_EQ(rd->shape[0], 3);
    CHECK_EQ(rd->shape[1], 3);
    CHECK_EQ(rd->data_type, FLOAT64);
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
    GraphNode *gn4 =
        createGraph(f3.data(), f3.size(), INT32, shape_f3.data(), 3);
    GraphNode *gn5 = add(gn4, result);
    result = executeGraph(gn5);
    rd = (ResultData *)result->operation;

    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++)
          CHECK_EQ(((double *)rd->data)[i * 9 + j * 3 + k],
                   v1[j][k] + v2[j][k] + v3[i][j][k]);
    freeGraph(result);
  }
}

int main(int argc, char **argv) {
  FlintBackend::init();
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  int res = context.run();
  FlintBackend::cleanup();
  if (context.shouldExit())
    return res;
  int client_stuff_return_code = 0;

  return res + client_stuff_return_code;
}
