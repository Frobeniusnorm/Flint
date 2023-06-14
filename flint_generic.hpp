#include "flint.h"
/**
 * Untyped version of tensors, minimal abstraction layer above `FGraphNode`.
 * They can only be initialized by a graph node,
 * and their data can only be collected through that graph node.
 * Only suggested for usage in cases were using the Tensor template does not
 * work anymore.
 */
struct GenericTensor {
  GenericTensor(FGraphNode *node) { node->reference_counter++; }
  /**
   * Copy constructor. Copies the underlying Graph structure by creating a new
   * node with the same operation, shape and data types. The new predecessor
   * array points to the same predecessors (memory safety is ensured with
   * reference counting).
   *
   * If `other` has result data or if it is a storage node, the complete CPU
   * data is directly copied. Since this operation is expensive it is advised to
   * only use it if it is completly necessary.
   */
  GenericTensor(const GenericTensor &other) {
    node = fCopyGraph(other.node);
    node->reference_counter++;
  }
  /**
   * Move constructor. Moves every important field from `other` to this Tensor.
   * `other` is invalidated after this operation.
   */
  GenericTensor(GenericTensor &&other) {
    node = other.node; // was held by previous tensor -> no increment necessary
    other.node = nullptr;
  }
  /**
   * Move operator. Moves every important field from `other` to this Tensor.
   * `other` is invalidated after this operation. If there was any previous
   * allocated operation node allocated by this Tensor it is cleaned up.
   */
  GenericTensor &operator=(GenericTensor &&other) {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
    node = other.node;
    other.node = nullptr;
    return *this;
  }
  /**
   * Copy operator. Copies the underlying Graph structure by creating a new
   * node with the same operation, shape and data types. If there was any
   * previous allocated operation node allocated by this Tensor it is cleaned
   * up. The new predecessor array points to the same predecessors (memory
   * safety is ensured with reference counting).
   *
   * If `other` has result data or if it is a storage node, the complete CPU
   * data is directly copied. Since this operation is expensive it is advised to
   * only use it if it is completly necessary.
   */
  GenericTensor &operator=(const GenericTensor &other) {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
    node = fCopyGraph(other.node);
    node->reference_counter++;
    return *this;
  }
  /**
   * Cleans up this tensor and frees all underlying data by reference counting.
   */
  ~GenericTensor() {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
  }

private:
  FGraphNode *node;
};
