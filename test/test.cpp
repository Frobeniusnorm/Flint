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
    GraphNode *gn1 = createGraph(v1.data(), v1.size(), FLOAT64);
    gn1 = add(gn1, 7.0);
    gn1 = mul(gn1, createGraph(v2.data(), v2.size(), FLOAT32));
    // test
    REQUIRE(gn1->num_predecessor == 2);
    GraphNode *right1 = gn1->predecessors[1];
    CHECK(right1->successor == gn1);
    CHECK(right1->num_predecessor == 0);
    CHECK(right1->predecessors == NULL);
    Store *store1 = (Store *)right1->operation;
    CHECK(store1->data_type == FLOAT32);
    CHECK(store1->num_entries == 100);
    CHECK(store1->data == v2.data());
    freeGraph(gn1);
  }
  {
    vector<long> v1(100);
    vector<int> v2(100);
    // construct graph 2
    GraphNode *gn2 = createGraph(v1.data(), v1.size(), INT64);
    gn2 = sub(gn2, 7.0);
    gn2 = div(gn2, createGraph(v2.data(), v2.size(), INT32));
    // test
    REQUIRE(gn2->num_predecessor == 2);
    GraphNode *right2 = gn2->predecessors[1];
    CHECK(right2->successor == gn2);
    CHECK(right2->num_predecessor == 0);
    CHECK(right2->predecessors == NULL);
    Store *store2 = (Store *)right2->operation;
    CHECK(store2->data_type == INT32);
    CHECK(store2->num_entries == 100);
    CHECK(store2->data == v2.data());
    freeGraph(gn2);
  }
}
TEST_CASE("Init and cleanup") {
  using namespace FlintBackend;
  init();
}
