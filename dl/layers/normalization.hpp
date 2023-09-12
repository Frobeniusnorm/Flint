#include "../layers.hpp"

/** Randomly sets some values in the input to 0 with a probability of `p`.
 * Reduces over fitting. Degenerates to an identity function when `training` is
 * false. */
class Dropout : public UntrainableLayer {
  double p;

public:
  static constexpr FType transform_type(FType t) { return F_FLOAT64; }
  Dropout(double p) : p(p) {}
  template <typename T, unsigned int n>
  Tensor<double, n> forward(Tensor<T, n> &in) {
    if (!training) {
      if constexpr (std::is_same<T, double>()) {
        return in;
      } else {
        return in.template convert<double>();
      }
    }
    Tensor<double, n> r = Flint::random_array(in.get_shape());
    // Tensor<double, n> r = Flint::constant_array(1.0, in.get_shape());
    Tensor<double, n> o = (in * (r > p)) / (1.0 - p);
    return o;
  }
  std::string name() override{
    return "Dropout";
  }
  std::string summary() override {
    return name() + " (p = " + std::to_string(p) + ")";
  }
};
