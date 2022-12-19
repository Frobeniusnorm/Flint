#ifndef GRADIENTS_CPP
#define GRADIENTS_CPP
#include "../flint.h"
static FGraphNode *constant_tensor(double val, FType type, size_t *shape,
                                   int dimensions) {
  switch (type) {
  case F_FLOAT32:
    return fconstant_f((float)val, shape, dimensions);
  case F_INT32:
    return fconstant_i((int)val, shape, dimensions);
  case F_INT64:
    return fconstant_l((long)val, shape, dimensions);
  case F_FLOAT64:
    return fconstant_d((double)val, shape, dimensions);
  }
}
FGraphNode *fgradient_add_g(const FGraphNode *x, const FGraphNode *y,
                            const FGraphNode *dx) {
  // ao is the higher dimensional vector
  const FOperation *ao = x->operation;
  const FOperation *bo = y->operation;
  const FGraphNode *a = x, *b = y;
  if (ao->dimensions < bo->dimensions) {
    const FOperation *tmp_op = ao;
    ao = bo;
    bo = tmp_op;
    const FGraphNode *tmp_gn = a;
    a = b;
    b = tmp_gn;
  }
  if (a == dx && b == dx) {
    // d(x + x = 2 * x) / dx = 2
    return constant_tensor(2.0, ao->data_type, ao->shape, ao->dimensions);
  } else if (b == dx) {
    // count number of times b is added to a and return tensor with shape
    // of b with that value
    int times = 1;
    for (int i = 0; i < ao->dimensions - bo->dimensions; i++)
      times *= ao->shape[i];
    return constant_tensor((double)times, bo->data_type, bo->shape,
                           bo->dimensions);
  } else if (a == dx) {
    // return one tensor with shape of a
    return constant_tensor(1.0, ao->data_type, ao->shape, ao->dimensions);
  } else {
    return constant_tensor(0.0, dx->operation->data_type, dx->operation->shape,
                           dx->operation->dimensions);
  }
}
FGraphNode *fgradient_sub_g(const FGraphNode *x, const FGraphNode *y,
                            const FGraphNode *dx) {
  // ao is the higher dimensional vector
  const FOperation *ao = x->operation;
  const FOperation *bo = y->operation;
  const FGraphNode *a = x, *b = y;
  bool switched = false;
  if (ao->dimensions < bo->dimensions) {
    const FOperation *tmp_op = ao;
    ao = bo;
    bo = tmp_op;
    const FGraphNode *tmp_gn = a;
    a = b;
    b = tmp_gn;
    switched = true;
  }
  if (a == dx && b == dx) {
    // d(x - x = x) / dx = 0
    return constant_tensor(0.0, ao->data_type, ao->shape, ao->dimensions);
  } else if (b == dx) {
    // count number of times b is added to a and return tensor with shape
    // of b with that value
    int times = 1;
    for (int i = 0; i < ao->dimensions - bo->dimensions; i++)
      times *= ao->shape[i];
    return constant_tensor((switched ? 1 : -1) * (double)times, bo->data_type,
                           bo->shape, bo->dimensions);
  } else if (a == dx) {
    // return one tensor with shape of a
    return constant_tensor((double)(switched ? -1 : 1), ao->data_type,
                           ao->shape, ao->dimensions);
  } else {
    return constant_tensor(0.0, dx->operation->data_type, dx->operation->shape,
                           dx->operation->dimensions);
  }
}
#endif
