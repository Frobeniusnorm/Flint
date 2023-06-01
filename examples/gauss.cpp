#include <flint/flint.h>
#include <flint/flint.hpp>

int main() {
  flintInit(FLINT_BACKEND_ONLY_CPU);
  fEnableEagerExecution();
  fSetLoggingLevel(F_INFO);
  Tensor<float, 3> img = Flint::load_image("flint.png");
  Tensor<float, 4> kernel {{{{1/16.0f}, {1/8.0f}, {1/16.0f}},
                            {{1/8.0f}, {1/4.0f}, {1/8.0f}},
                            {{1/16.0f}, {1/8.0f}, {1/16.0f}}}};
  for (size_t s : img.get_shape()) std::cout << s << std::endl;
  //img = img.slice(TensorRange(TensorRange::MAX_SCOPE, TensorRange::MAX_SCOPE, -2));
  size_t h = img.get_shape()[0], w = img.get_shape()[1], c = img.get_shape()[2];
  // put channels in first dimension
  img = img.transpose();
  for (int i = 0; i < 100; i++) {
    // add left padding
    img = img.extend({c, w + 3, h + 3}, {0, 3, 3});
    // gauss
    img = img.reshape(c, w + 3, h + 3, 1).convolve(kernel, 1, 1, 1);
    // undo padding
    img = img.slice(TensorRange(), TensorRange(2, -2), TensorRange(2, -2));
    img.execute();
  }
  // undo transpose
  img = img.transpose();
  img.execute();
  flogging(F_INFO, "executed");
  for (size_t s : img.get_shape()) std::cout << s << std::endl;
  Flint::store_image(img, "experiment/flint.jpg", F_JPEG);
  flintCleanup();
}
