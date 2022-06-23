#include "../flint.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include <stdlib.h>
#include <unordered_set>
#include <vector>
using namespace FlintBackend;

GraphNode *createGraph(void *data, int num_entries, Type data_type) {
  GraphNode *gn = new GraphNode();
  Store *op = new Store();
  op->data = data;
  op->num_entries = num_entries;
  op->data_type = data_type;
  gn->operation = op;
  gn->num_predecessor = 0;
  gn->predecessors = NULL;
  gn->successor = NULL;
  return gn;
}
static void collectAllNodes(GraphNode *graph,
                            std::unordered_set<GraphNode *> &set) {
  set.insert(graph);
  if (graph->successor && set.find(graph->successor) == set.end()) {
    collectAllNodes(graph->successor, set);
  }
  for (int i = 0; i < graph->num_predecessor; i++) {
    if (set.find(graph->predecessors[i]) == set.end())
      collectAllNodes(graph->predecessors[i], set);
  }
}
// frees all allocated data from the graph and the nodes that are reachable
void freeGraph(GraphNode *graph) {
  std::unordered_set<GraphNode *> all;
  collectAllNodes(graph, all);
  for (GraphNode *gn : all) {
    delete gn->operation;
    if (gn->predecessors)
      free(gn->predecessors);
    delete gn;
  }
}
// function to add nodes to the graph i.e. operations
static GraphNode *addNode(Operation *op, std::vector<GraphNode *> pre) {
  GraphNode *foo = new GraphNode();
  foo->operation = op;
  foo->num_predecessor = pre.size();
  foo->predecessors = safe_mal<GraphNode *>(pre.size());
  for (int i = 0; i < pre.size(); i++) {
    foo->predecessors[i] = pre[i];
    pre[i]->successor = foo;
  }
  return foo;
}

GraphNode *add(GraphNode *a, GraphNode *b) {
  Add *op = new Add();
  return addNode(op, {a, b});
}
GraphNode *sub(GraphNode *a, GraphNode *b) {
  Sub *op = new Sub();
  return addNode(op, {a, b});
}
GraphNode *div(GraphNode *a, GraphNode *b) {
  Div *op = new Div();
  return addNode(op, {a, b});
}
GraphNode *mul(GraphNode *a, GraphNode *b) {
  Mul *op = new Mul();
  return addNode(op, {a, b});
}

// adds the constant value to each entry in a
template <typename T> GraphNode *add(GraphNode *a, T b) {
  Add *op = new Add();
  return addNode(op, {a, new Const(b)});
}
// subtracts the constant value from each entry in a
template <typename T> GraphNode *sub(GraphNode *a, T b);
// divides each entry in a by the constant value
template <typename T> GraphNode *div(GraphNode *a, T b);
// multiplicates the constant value with each entry in a
template <typename T> GraphNode *mul(GraphNode *a, T b);
