#include <flint/flint.h>
#include <flint/flint.hpp>

int main() {
  FlintContext _(FLINT_BACKEND_ONLY_GPU);
  fSetLoggingLevel(F_INFO);
  Tensor<float, 3> img = Flint::load_image("../../flint.png");
  Tensor<float, 4> kernel{{{{1 / 16.0f}, {1 / 8.0f}, {1 / 16.0f}},
                           {{1 / 8.0f}, {1 / 4.0f}, {1 / 8.0f}},
                           {{1 / 16.0f}, {1 / 8.0f}, {1 / 16.0f}}}};
  size_t h = img.get_shape()[0], w = img.get_shape()[1], c = img.get_shape()[2];
  // put channels in first dimension
  img = img.transpose();
  for (int i = 0; i < 500; i++) {
    // add left and right padding
    img = img.extend({c, w + 2, h + 2}, {0, 1, 1});
    // gauss
    img = img.reshape(c, w + 2, h + 2, 1).convolve(kernel, 1, 1, 1);
    img.execute();
  }
  // undo transpose
  img = img.transpose();
  Flint::store_image(img, "flint.jpg", F_JPEG);
}
