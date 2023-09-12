#include "../layers.hpp"
/**
 * Padding of convolution operations
 * - `NO_PADDING`: a normal convolution operation. Each filter is slid over the
 *   input tensor with its step size as many times as it completly fits into the
 *   input tensor. The output may have a smaller size then the input.
 */
enum PaddingMode { NO_PADDING };
/**
 * A generic Convolution layer. It creates multiple filters that are slid along
 * the input in each dimension by a step size. Each time the filter values are
 * multiplied with the elements of the input tensor with which it is currently
 * aligned and the result (with shape of the filter) is summed up to a single
 * value in the resulting tensor. After that the filter is moved by its step
 * size in each dimension and the process repeats.
 *
 * TLDR; Each element in the result of this layer is a full multiplication of a
 * filter with a corresponding window in the input array. This is especially
 * helpful for image processing tasks, since the parameters (filters) allow the
 * recognize location independent features in the input.
 *
 * You are supposed to configure this layer by providing a number of `filters`,
 * a `kernel_size` and the size of the last dimension of the input tensor, i.e.
 * the channels of the input tensor called `units_in`. The template expects you
 * to provide the dimensionality of the input tensor (including batch size and
 * channels). The output size is the same as of the input tensor for the first
 * dimension (usually the `batch_size`), in the last dimension it is the number
 * of `filters` and in every other the number of times each filter can be slid
 * against the input tensor (depending on the size of the input tensor, the
 * `kernel_size`, the step size and padding see `PaddingMode`).
 *
 * E.g. if you have a batch of two dimensional rgb (3 channels) images, it
 * would have a shape of `(batch_size, height, width, 3)`. Then you would
 * create a `Convolution<4>` layer (also called `Conv2D`) with `units_in = 3`.
 * The output tensor would also be a 4 dimensional tensor.
 * Lets say you dont use padding (`NO_PADDING`), 10 filters, a step size of 2 in
 * each dimension, 32 as `kernel_size` and your 100 images have widths and
 * heights of `128` (`input_shape = (100, 128, 128, 3)`).
 * The output size would be
 * `(batch_size, ceil((input_shape - kernel_size + 1) / steps),
 *   ceil((input_shape - kernel_size + 1) / steps), filters) =
 *   (100, 49, 49, 10)`.
 */
template <int n> class Convolution : public Layer<n> {
  constexpr std::array<size_t, n> weight_shape(unsigned int filters,
                                               unsigned int kernel_size,
                                               size_t units_in) {
    std::array<size_t, n> res;
    res[0] = filters;
    for (int i = 1; i < n - 1; i++) {
      res[i] = kernel_size;
    }
    res[n - 1] = units_in;
    return res;
  }
  std::array<unsigned int, n - 1> act_stride;
  unsigned int kernel_size;
  void initialize_precalc(std::array<unsigned int, n - 2> stride) {
    act_stride[0] = 1;
    for (int i = 0; i < n - 2; i++)
      act_stride[i + 1] = stride[i];
  }

public:
  PaddingMode padding_mode;
  /** Initializes the Convolution Layer.
   * - `units_in` number of channels (size of last dimension) of input tensor
   * - `filters` number of used filters (size of last dimension of the result
   *    tensor)
   * - `kernel_size` size of filters
   * - `init` Initializer for filters, has to implement the `Initializer`
   *    concept, should generate random values close to a normal distribution
   * - `stride` step size per dimension (2 dimensions less then the input
   *    tensor, since the convolution is broadcasted along the `batch_size` and the
   *    channels in the last dimension are fully reduced)
   * - `padding_mode` which type of padding to use (see `PaddingMode` for more
   *    information)
   */
  template <Initializer InitWeights>
  Convolution(size_t units_in, unsigned int filters, unsigned int kernel_size,
              InitWeights init, std::array<unsigned int, n - 2> stride,
              PaddingMode padding_mode = NO_PADDING)
      : Layer<n>(init.template initialize<double>(
            weight_shape(filters, kernel_size, units_in))),
        padding_mode(padding_mode), kernel_size(kernel_size) {
    initialize_precalc(stride);
  }

  /** Initializes the Convolution Layer.
   * - `units_in` number of channels (size of last dimension) of input tensor
   * - `filters` number of used filters (size of last dimension of the result
   *    tensor)
   * - `kernel_size` size of filters
   * - `stride` step size per dimension (2 dimensions less then the input
   *    tensor, since the convolution is broadcasted along the `batch_size` and the
   *    channels in the last dimension are fully reduced)
   * - `padding_mode` which type of padding to use (see `PaddingMode` for more
   *    information)
   *
   * The filters are initialized with a glorot uniform distribution.
   */
  Convolution(size_t units_in, unsigned int filters, unsigned int kernel_size,
              std::array<unsigned int, n - 2> stride,
              PaddingMode padding_mode = NO_PADDING)
      : Layer<n>(GlorotUniform().template initialize<double>(
            weight_shape(filters, kernel_size, units_in))),
        padding_mode(padding_mode), kernel_size(kernel_size) {

    initialize_precalc(stride);
  }
  std::string name() override { return "Convolution"; }
  std::string summary() override {
    const unsigned int filters =
        Layer<n>::template get_weight<0>().get_shape()[0];
    const unsigned int units_in =
        Layer<n>::template get_weight<0>().get_shape()[n - 1];
    const unsigned int kernel_size =
        Layer<n>::template get_weight<0>().get_shape()[1];
    return name() + ": input channels: " + std::to_string(units_in) +
           " filters: " + std::to_string(filters) +
           ", kernel size: " + std::to_string(kernel_size);
  }
  template <typename T, unsigned int k>
  Tensor<double, k> forward(Tensor<T, k> &in) {
    const unsigned int filters =
        Layer<n>::template get_weight<0>().get_shape()[0];
    // actual convolve
    Tensor<double, k> res;
    for (unsigned int i = 0; i < filters; i++) {
      // in has shape [batch, dim1, ..., units_in]
      // has shape [1, kernel_size, ..., units_in]
      Tensor<double, n> filter =
          Layer<n>::template get_weight<0>().slice(TensorRange(i, i + 1));
      Tensor<double, n - 1> filter_res = in.convolve_array(filter, act_stride);
      std::array<size_t, n> new_shape;
      for (int i = 0; i < n - 1; i++)
        new_shape[i] = filter_res.get_shape()[i];
      new_shape[n - 1] = 1;
      Tensor<double, k> local_res = filter_res.reshape_array(new_shape);
      local_res.execute();
      res = i == 0 ? local_res : Flint::concat(res, local_res, n - 1);
    }
    return res;
  }
};
/** For inputs of images with shape `(batch_size, width, height, channels)` */
typedef Convolution<4> Conv2D; 
                               
