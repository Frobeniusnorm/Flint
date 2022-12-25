#include "../flint.h"
#include "../flint.hpp"
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

TEST_SUITE("Gradient Calculation") {
  TEST_CASE("Gradient add/sub") {
    Tensor<float, 2> t1{{-1., 0.}, {1., 2.}};
    Tensor<double, 3> t2{{{0.0, 1.0}, {2.0, 3.0}}, {{4.0, 5.0}, {6.0, 7.0}}};
    FGraphNode *res = fgradient_add(t1.get_graph_node(), t2.get_graph_node(),
                                    t1.get_graph_node());
    FOperation *res_op = res->operation;
    CHECK_EQ(F_FLOAT32, res_op->data_type);
    CHECK_EQ(2, res_op->dimensions);
    CHECK_EQ(2, res_op->shape[0]);
    CHECK_EQ(2, res_op->shape[1]);
    res = fExecuteGraph(res);
    res_op = res->operation;
    FStore *store = (FStore *)res_op->additional_data;
    float *dataf = (float *)store->data;
    for (int i = 0; i < 4; i++)
      CHECK_EQ(2.f, dataf[i]);
    fFreeGraph(res);
    res = fgradient_sub(t1.get_graph_node(), t2.get_graph_node(),
                        t1.get_graph_node());
    res = fExecuteGraph(res);
    store = (FStore *)res->operation->additional_data;
    dataf = (float *)store->data;
    for (int i = 0; i < 4; i++)
      CHECK_EQ(2.f, dataf[i]);
    fFreeGraph(res);
  }
  TEST_CASE("Gradient mul") {
    Tensor<float, 2> t1{{-1., 0.}, {1., 2.}};
    Tensor<double, 3> t2{{{0.0, 1.0}, {2.0, 3.0}}, {{4.0, 5.0}, {6.0, 7.0}}};
    FGraphNode *res = fgradient_mul(t1.get_graph_node(), t2.get_graph_node(),
                                    t1.get_graph_node());
    CHECK_EQ(2, res->operation->dimensions);
    res = fExecuteGraph(res);
    FStore *store = (FStore *)res->operation->additional_data;
    double *datad = (double *)store->data;
    CHECK_EQ(4, datad[0]);
    CHECK_EQ(6, datad[1]);
    CHECK_EQ(8, datad[2]);
    CHECK_EQ(10, datad[3]);
    // fFreeGraph(res);
    res = fgradient_mul(t1.get_graph_node(), t2.get_graph_node(),
                        t2.get_graph_node());
    CHECK_EQ(3, res->operation->dimensions);
    res = fExecuteGraph(res);
    store = (FStore *)res->operation->additional_data;
    float *dataf = (float *)store->data;
    CHECK_EQ(-1, dataf[0]);
    CHECK_EQ(0, dataf[1]);
    CHECK_EQ(1, dataf[2]);
    CHECK_EQ(2, dataf[3]);
    CHECK_EQ(-1, dataf[4]);
    CHECK_EQ(0, dataf[5]);
    CHECK_EQ(1, dataf[6]);
    CHECK_EQ(2, dataf[7]);
    fFreeGraph(res);
  }
  TEST_CASE("Gradient div") {
    Tensor<double, 2> x = {{-1., 3.}, {1., 2.}};
    Tensor<double, 3> y = {{{1.0, 1.0}, {2.0, 3.0}}, {{4.0, 5.0}, {6.0, 7.0}}};

    FGraphNode *node = fgradient_div(y.get_graph_node(), x.get_graph_node(),
                                     x.get_graph_node());
    FGraphNode *res = fExecuteGraph(node);
    FResultData *data = (FResultData *)res->operation->additional_data;
    double *datad = (double *)data->data;
    CHECK_EQ(-5, datad[0]);
    CHECK_EQ(-8, datad[2]);
    CHECK_EQ(-2.5, datad[3]);
    fFreeGraph(res);
    node = fgradient_div(y.get_graph_node(), x.get_graph_node(),
                         y.get_graph_node());
    res = fExecuteGraph(node);
    data = (FResultData *)res->operation->additional_data;
    datad = (double *)data->data;
    CHECK_EQ(-1, datad[0]);
    CHECK_EQ(1, datad[2]);
    CHECK_EQ(.5, datad[3]);
    fFreeGraph(res);
    node = fgradient_div(x.get_graph_node(), y.get_graph_node(),
                         y.get_graph_node());
    res = fExecuteGraph(node);
    data = (FResultData *)res->operation->additional_data;
    datad = (double *)data->data;
    CHECK_EQ(1, datad[0]);
    CHECK_EQ(-3, datad[1]);
    CHECK_EQ(-.12, datad[5]);
    fFreeGraph(res);
    node = fgradient_div(x.get_graph_node(), y.get_graph_node(),
                         x.get_graph_node());
    res = fExecuteGraph(node);
    data = (FResultData *)res->operation->additional_data;
    datad = (double *)data->data;
    CHECK_EQ(1.25, datad[0]);
    CHECK_EQ(1.2, datad[1]);
    fFreeGraph(res);
  }
}
int main(int argc, char **argv) {
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  flintInit(1, 0);
  int res = context.run();
  flintCleanup();
  return res;
}
