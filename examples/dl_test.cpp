#include "../dl/layers.hpp"
#include "../dl/models.hpp"
#include <flint/flint.h>
#include <flint/flint.hpp>
#include <flint/flint_helper.hpp>
int main() {
  flintInit(FLINT_BACKEND_ONLY_GPU);
  Flint::setLoggingLevel(F_DEBUG);
  Tensor<float, 3> t1{{{0, 1}, {1, 2}, {3, 4}},
                      {{5, 6}, {7, 8}, {9, 0}},
                      {{-1, -2}, {-3, -4}, {-5, -6}}};
  auto m =
      SequentialModel<Connected, Connected>{Connected(2, 4), Connected(4, 2)};
  AdamFactory adam(0.1);
  m.generate_optimizer(&adam);
  for (int i = 0; i < 100; i++) {
    fStartGradientContext();
    Tensor<double, 3> o =
        ((t1 + (t1.slice(TensorRange(0, 1), TensorRange(0, 1), TensorRange(TensorRange::MAX_SCOPE, TensorRange::MAX_SCOPE, -1)).flattened())) -
         m.forward(t1))
            .abs();
    fStopGradientContext();
    Tensor<double, 1> e = o.reduce_sum(0).reduce_sum(0).reduce_sum()();
    std::cout << e << std::endl;
    m.optimize(o);
  }
  flintCleanup();
}
