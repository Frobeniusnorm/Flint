#include "../dl/layers.hpp"
#include "../dl/activations.hpp"
#include "../dl/models.hpp"
#include <flint/flint.h>
#include <flint/flint.hpp>
#include <flint/flint_helper.hpp>
int main() {
  FlintContext _(FLINT_BACKEND_ONLY_CPU);
  Flint::setLoggingLevel(F_VERBOSE);
  Tensor<float, 3> t1{{{0, 1}, {1, 2}, {3, 4}},
                      {{5, 6}, {7, 8}, {9, 0}},
                      {{-1, -2}, {-3, -4}, {-5, -6}}};
  auto m = SequentialModel{
    Flatten(),
    Connected(18, 32),
    Relu(),
    Dropout(0.2),
    Connected(32, 10),
    SoftMax()
  };
  AdamFactory adam (0.05);
  m.generate_optimizer(&adam);
  // TODO load data m.train(X, Y, CrossEntropyLoss(), 100);
}
