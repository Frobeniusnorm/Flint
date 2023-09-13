
#include "../layers.hpp"

struct Flatten : public UntrainableLayer {
  static constexpr unsigned int transform_dimensionality(unsigned int n) {
    return 2;
  }
  /** Flattens every feature axis into one single axis, does not touch the
   * batch-axis (the first) */
  template <typename T, unsigned int n>
  Tensor<T, 2> forward(const Tensor<T, n> &in) {
    if constexpr (n == 2)
      return Tensor<T, 2>(in);
    else
      return forward(in.flattened(n - 1));
  }
  std::string name() override {
    return "Flatten";
  }
};
