#include "../flint.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include <stdlib.h>
#include <unordered_set>
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
    if (set.find(&graph->predecessors[i]) == set.end())
      collectAllNodes(&graph->predecessors[i], set);
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
