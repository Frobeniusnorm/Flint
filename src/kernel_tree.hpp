#ifndef KERNEL_TREE
#define KERNEL_TREE
#include "../flint.h"
#include <CL/cl.h>
#include <array>
#include <vector>
namespace KernelTree {
struct TreeNode {
  int id;
  cl_program program;
  cl_kernel kernel;
  std::array<TreeNode *, sizeof(FOperationType)> children;
  TreeNode() {
    program = nullptr;
    kernel = nullptr;
    for (int i = 0; i < FOperationType::Length; i++)
      children[i] = nullptr;
  }
  ~TreeNode() {
    if (kernel != nullptr)
      clReleaseKernel(kernel);
    if (program)
      clReleaseProgram(program);
  }
};
static std::array<TreeNode *, FOperationType::Length> *roots;
static std::vector<TreeNode *> *allNodes;
inline void kernelTreeInit() {
  roots = new std::array<TreeNode *, FOperationType::Length>();
  for (int i = 0; i < FOperationType::Length; i++)
    (*roots)[i] = nullptr;
  allNodes = new std::vector<TreeNode *>();
}
static void cleanUpNode(TreeNode *node) {
  for (TreeNode *child : node->children) {
    cleanUpNode(child);
    delete child;
  }
}
inline void kernelTreeCleanUp() {
  for (TreeNode *root : *roots) {
    cleanUpNode(root);
    delete root;
  }
  free(roots);
}

// goes down the edge corresponding to operation
inline int kernelStepDown(int curr, FOperationType operation) {
  TreeNode *currNode = (*allNodes)[curr];
  if (!currNode->children[operation]) {
    TreeNode *newNode = new TreeNode();
    newNode->id = allNodes->size();
    allNodes->push_back(newNode);
    currNode->children[operation] = newNode;
  }
  return currNode->children[operation]->id;
}
inline cl_kernel getKernel(int curr) { return (*allNodes)[curr]->kernel; }
inline void storeKernel(int curr, cl_kernel kernel, cl_program program) {
  TreeNode *node = (*allNodes)[curr];
  node->kernel = kernel;
  node->program = program;
}
}; // namespace KernelTree
#endif
