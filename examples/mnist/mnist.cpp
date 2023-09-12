// #define FLINT_DL_PROFILE
#include <flint/dl/flint_dl.hpp>
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
static Tensor<float, 3> load_mnist_images(const std::string path) {
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
    std::vector<float> data(no * h * w);
    for (int i = 0; i < no; i++) {
      for (int j = 0; j < h; j++) {
        for (int k = 0; k < w; k++) {
          unsigned char value;
          file.read((char *)&value, 1);
          data[i * h * w + j * w + k] = (float)value / 255.0;
        }
      }
    }
    std::array<size_t, 3> shape{(size_t)no, (size_t)h, (size_t)w};
    return Tensor<float, 3>(fCreateGraph(data.data(), no * h * w, F_FLOAT32,
                                         shape.data(), shape.size()),
                            shape);
  } else
    throw std::runtime_error("Could not load file! Please download it from "
                             "http://yann.lecun.com/exdb/mnist/");
}
static Tensor<int, 2> load_mnist_labels(const std::string path) {
  using namespace std;
  errno = 0;
  ifstream file(path);
  if (file.is_open()) {
    int magic_number = 0;
    int no = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char *)&no, sizeof(no));
    no = reverseInt(no);
    std::vector<int> data(no * 10);
    for (int i = 0; i < no; i++) {
      unsigned char value;
      file.read((char *)&value, 1);
      for (int j = 0; j < 10; j++) {
        data[i * 10 + j] = value == j ? 1 : 0;
      }
    }
    std::array<size_t, 2> shape{(size_t)no, 10};
    return Tensor<int, 2>(
        fCreateGraph(data.data(), no * 10, F_INT32, shape.data(), 2), shape);
  } else
    throw std::runtime_error("Could not load file! Please download it from "
                             "http://yann.lecun.com/exdb/mnist/");
}

// download and extract to the desired folder from
// http://yann.lecun.com/exdb/mnist/
int main() {
  FlintContext _(FLINT_BACKEND_BOTH);
  fSetLoggingLevel(F_INFO);
  Tensor<float, 3> X = load_mnist_images("train-images-idx3-ubyte");
  Tensor<double, 2> Y =
      load_mnist_labels("train-labels-idx1-ubyte").convert<double>();
  Tensor<float, 3> vX = load_mnist_images("t10k-images-idx3-ubyte");
  Tensor<double, 2> vY =
      load_mnist_labels("t10k-labels-idx1-ubyte").convert<double>();
  auto data = TrainingData(
      X.reshape(X.get_shape()[0], X.get_shape()[1], X.get_shape()[2], 1), 
      Y,
      vX.reshape(vX.get_shape()[0], vX.get_shape()[1], vX.get_shape()[2], 1),
      vY
      );
  std::cout << data.X.get_shape()[0] << " images à " << data.X.get_shape()[1]
            << "x" << data.X.get_shape()[1] << " (and " << data.Y.get_shape()[0]
            << " labels)" << std::endl;
  std::cout << "loaded data. Starting training." << std::endl;
  auto m = SequentialModel{
    Conv2D(1, 10, 7, std::array<unsigned int, 2>{2, 2}, NO_PADDING),
    Relu(),
    Flatten(),
    Connected(1210, 80),
    Relu(),
    Connected(80, 10),
    SoftMax()
  };
  std::cout << m.summary() << std::endl;
  AdamFactory opt(0.003);
  m.generate_optimizer(opt);
  m.train(data, CrossEntropyLoss(), 75, 4000);
}
