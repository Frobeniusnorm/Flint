#include "../../dl/activations.hpp"
#include "../../dl/layers.hpp"
#include "../../dl/models.hpp"
#include <cstring>
#include <flint/flint.h>
#include <flint/flint.hpp>
#include <flint/flint_helper.hpp>
#include <stdexcept>

int reverseInt(int i) {
  unsigned char c1, c2, c3, c4;
  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
static Tensor<double, 3> load_mnist_images(const std::string path) {
  using namespace std;
  errno = 0;
  ifstream file(path);
  if (file.is_open()) {
    int magic_number = 0;
    int no = 0;
    int h = 0;
    int w = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char *)&no, sizeof(no));
    no = reverseInt(no);
    file.read((char *)&h, sizeof(h));
    h = reverseInt(h);
    file.read((char *)&w, sizeof(w));
    w = reverseInt(w);
    std::vector<double> data(no * h * w);
    for (int i = 0; i < no; i++) {
      for (int j = 0; j < h; j++) {
        for (int k = 0; k < w; k++) {
          unsigned char value;
          file.read((char *)&value, 1);
          data[i * h * w + j * w + k] = value / 255.0;
        }
      }
    }
    std::array<size_t, 3> shape{(size_t)no, (size_t)h, (size_t)w};
    return Tensor<double, 3>(fCreateGraph(data.data(), no * h * w, F_FLOAT64,
                                         shape.data(), shape.size()),
                            shape);
  } else
    throw std::runtime_error("Could not load file! Please download it from "
                             "http://yann.lecun.com/exdb/mnist/");
}
static Tensor<int, 2> load_mnist_labels(const std::string path) {}

// download and extract to the desired folder from
// http://yann.lecun.com/exdb/mnist/
int main() {
  FlintContext _(FLINT_BACKEND_ONLY_CPU);
  read_mnist("train-images.idx3-ubyte");
  Tensor<double, 3> ims = load_mnist_images("train-images.idx3-ubyte");
}
