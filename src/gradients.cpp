#ifndef GRADIENTS_CPP
#define GRADIENTS_CPP
#include "../flint.h"

FGraphNode *fgradient_add_g(const FGraphNode *a, const FGraphNode *b,
                            const FGraphNode *dx) {
  // ao is the higher dimensional vector
  const FOperation *ao = a->operation;
  const FOperation *bo = b->operation;
  if (ao->dimensions < bo->dimensions) {
    const FOperation *tmp = ao;
    ao = bo;
    bo = tmp;
  }
  if (b == dx) {
    // TODO count number of times b is added to a and return tensor with shape
    // of b with that value
  } else if (a == dx) {
    // TODO return one tensor with shape of a
  } else {
    // TODO return zero tensor with shape of dx
  }
  return nullptr;
}

#endif
