#include "../flint.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <vector>
TEST_CASE("Graph Implementation (createGraph, add, mul, sub, div)") {
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
    gn1 = mul(gn1, createGraph(v2.data(), v2.size(), FLOAT32, shape.data(), 1));
    // test
    REQUIRE_EQ(gn1->num_predecessor, 2);
    REQUIRE(gn1->operation);
    CHECK_EQ(gn1->operation->data_type, FLOAT64);
    GraphNode *right1 = gn1->predecessors[1];
    CHECK_EQ(right1->num_predecessor, 0);
    CHECK_EQ(right1->predecessors, NULL);
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
    GraphNode *gn2 = createGraph(v1.data(), v1.size(), INT64, shape.data(), 2);
    gn2 = sub(gn2, 7.0);
    gn2 = div(gn2, createGraph(v2.data(), v2.size(), INT32, shape.data(), 2));
    // test
    REQUIRE(gn2->num_predecessor == 2);
    REQUIRE(gn2->operation);
    CHECK(gn2->operation->op_type == DIV);
    CHECK(gn2->operation->data_type == INT64);
    GraphNode *right2 = gn2->predecessors[1];
    CHECK(right2->num_predecessor == 0);
    CHECK(right2->predecessors == NULL);
    REQUIRE(gn2->operation);
    Store *store2 = (Store *)right2->operation;
    CHECK(store2->data_type == INT32);
    CHECK(store2->num_entries == 100);
    CHECK(store2->data == v2.data());
    GraphNode *left1 = gn2->predecessors[0];
    GraphNode *const1 = left1->predecessors[1];
    CHECK(const1->operation->op_type == CONST);
    freeGraph(gn2);
  }
}
TEST_CASE("Init, Execution and cleanup") {
  using namespace FlintBackend;
  using namespace std;
  init();
  vector<double> v1(10, 4.0);
  vector<float> v2(10, 4.0f);
  // construct graph 1
  vector<int> shape{10};
  GraphNode *gn1 = createGraph(v1.data(), v1.size(), FLOAT64, shape.data(), 1);
  gn1 = add(gn1, 7.0);
  gn1 = mul(gn1, createGraph(v2.data(), v2.size(), FLOAT32, shape.data(), 1));
  GraphNode *result = executeGraph(gn1);
  ResultData *rd = (ResultData *)result->operation;
  CHECK(rd->num_entries == 10);
  for (int i = 0; i < rd->num_entries; i++)
    CHECK(((double *)rd->data)[i] == 44);
  // construct graph 2 (first not-tree)
  vector<float> v3(10);
  for (int i = 0; i < 10; i++)
    v3[i] = i + 1;
  GraphNode *gn2 = createGraph(v3.data(), v3.size(), FLOAT32, shape.data(), 1);
  GraphNode *gn3 = add(gn2, result);
  gn3 = add(gn3, result);
  gn3 = sub(gn3, 80);
  gn3 = add(gn3, gn2);
  result = executeGraph(gn3);
  rd = (ResultData *)result->operation;
  CHECK(rd->num_entries == 10);
  for (int i = 0; i < 10; i++)
    CHECK(((double *)rd->data)[i] == 8 + (i + 1) * 2);
  freeGraph(result);
  cleanup();
}
