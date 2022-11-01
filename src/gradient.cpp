#include "../flint_nn.h"
/*
 * Implement the gradient calculation with either automatic differentiation or
 * corresponding graph functions. Forumlas:
 * gradient_e(i) = e(i) * f(i)'(y) * x(i)
 * e(i-1) = e(i) * f(i)'(y) * w(i)
 */
FGradientRecord *fcalculateGradients(FGraphNode *y, FGraphNode *error,
                                     FGraphNode **vars_to_watch, int num_vars) {
  // TODO
  return nullptr;
}
FGraphNode *fgetErrorGradient(FGradientRecord *record, FGraphNode *variable) {
  // TODO
  return nullptr;
}
