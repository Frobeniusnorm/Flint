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
    CHECK_EQ(right1->successor, gn1);
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
    CHECK(right2->successor == gn2);
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
  vector<double> v1(100);
  vector<float> v2(100);
  // construct graph 1
  vector<int> shape{100};
  GraphNode *gn1 = createGraph(v1.data(), v1.size(), FLOAT64, shape.data(), 1);
  gn1 = add(gn1, 7.0);
  gn1 = mul(gn1, createGraph(v2.data(), v2.size(), FLOAT32, shape.data(), 1));
  executeGraph(gn1);
  cleanup();
}
