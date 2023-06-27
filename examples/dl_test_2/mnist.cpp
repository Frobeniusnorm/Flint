#include "../../dl/activations.hpp"
#include "../../dl/layers.hpp"
#include "../../dl/models.hpp"
#include <flint/flint.h>
#include <flint/flint.hpp>
#include <flint/flint_helper.hpp>

static int read_int_at_pos(char *buff, int pos) {
  return (buff[pos] << 3) & (buff[pos + 1] << 2) & (buff[pos + 2] << 1) &
         buff[pos + 3];
}

static Tensor<float, 3> load_mnist_images(const std::string path) {
  char *buffer;
  long size;
  using namespace std;
  ifstream file("train-images.idx3-ubyte", ios::in | ios::binary | ios::ate);
  size = file.tellg();
  std::cout << size << std::endl;
  file.seekg(0, ios::beg);
  buffer = new char[size];
  file.read(buffer, size);
  file.close();
  unsigned int magic = read_int_at_pos(&buffer[0], 0);
  std::cout << magic << std::endl;
  unsigned int no = read_int_at_pos(buffer, 4);
  int h = read_int_at_pos(buffer, 8), w = read_int_at_pos(buffer, 12);
  std::cout << no << ", " << h << ", " << w << std::endl;
  std::vector<float> data(no * h * w);
  for (int i = 0; i < no; i++) {
    for (int j = 0; j < h; j++) {
      for (int k = 0; k < w; k++) {
        char value = buffer[12 + i * h * w + j * w + i];
        data[i * h * w + j * w + i] = value / 255.0;
        std::cout << value << std::endl;
      }
    }
  }
  delete buffer;
  std::array<size_t, 3> shape{(size_t)no, (size_t)h, (size_t)w};
  return Tensor<float, 3>(fCreateGraph(data.data(), no * h * w, F_FLOAT64,
                                       shape.data(), shape.size()),
                          shape);
}
static Tensor<int, 2> load_mnist_labels(const std::string path) {}

// download and extract to the desired folder from
// http://yann.lecun.com/exdb/mnist/
int main() {
  FlintContext _;
  Tensor<float, 3> ims = load_mnist_images("train-images.idx3-ubyte");
  Tensor<float, 3> fst_img =
      ims.slice(TensorRange(0, 1)).flattened(0).expand(3, 3);
  Flint::store_image(fst_img, "tst-img.png", FImageFormat::F_PNG);
}
