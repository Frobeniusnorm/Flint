#ifndef ONNX_LAYERS
#define ONNX_LAYERS
#include "onnx.proto3.pb.h"
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#define FLINT_DEBUG
#include <flint/flint.h>
#include <vector>
static void compute_fans(std::vector<size_t> shape, unsigned int &fan_in,
						 unsigned int &fan_out) {
	const int n = shape.size();
	if (n == 1) {
		fan_in = shape[0];
		fan_out = shape[0];
	} else if (n == 2) {
		fan_in = shape[0];
		fan_out = shape[1];
	} else {
		size_t acc = 1;
		for (int i = 0; i < n - 2; i++)
			acc *= shape[i];
		fan_in = shape[n - 2] * acc;
		fan_out = shape[n - 1] * acc;
	}
}
// TODO adapt the data s.t. channels are switched to back to remove the
// transpositions
/** Base class for all layer graph nodes. */
struct LayerGraph {
		std::vector<LayerGraph *> incoming; // incoming edges in graph
		std::vector<LayerGraph *> outgoing; // incoming edges in graph
		std::vector<FGraphNode *> output;	// result of forward
		bool training = false;
		std::string name;
		LayerGraph() = default;
		virtual ~LayerGraph() {
			marked_for_deletion = true;
			for (LayerGraph *lg : incoming)
				if (lg && !lg->marked_for_deletion) {
					lg->marked_for_deletion = true;
					delete lg;
				}
			for (LayerGraph *lg : outgoing)
				if (lg && !lg->marked_for_deletion) {
					lg->marked_for_deletion = true;
					delete lg;
				}
		}
		LayerGraph(size_t reserved_output_slots)
			: output(reserved_output_slots) {
			for (size_t i = 0; i < reserved_output_slots; i++)
				output[i] = nullptr;
		}
		LayerGraph(LayerGraph *in, size_t reserved = 1) : LayerGraph(reserved) {
			incoming.push_back(in);
			in->outgoing.push_back(this);
		}
		LayerGraph(std::vector<LayerGraph *> in, size_t reserved = 1)
			: LayerGraph(reserved) {
			for (LayerGraph *i : in) {
				if (i) {
					incoming.push_back(i);
					i->outgoing.push_back(this);
				}
			}
		}
		/**
		 * Computes the layer output in `output` by the output of the
		 * previous layers. The framework makes sure that the output of
		 * the incoming layers exists.
		 */
		virtual void forward() = 0;
		virtual void deserialize_to_onnx(onnx::NodeProto *) {
			flogging(F_WARNING,
					 "Node is deserialized to onnx without an implementation.");
		};
		virtual std::vector<std::vector<size_t>>
		propagate_shape(const std::vector<std::vector<size_t>> &input) {
			std::vector<std::vector<size_t>> out(output.size());
			for (int i = 0; i < out.size(); i++)
				out[i] = input[i];
			return out;
		}

	private:
		bool marked_for_deletion = false; // for destructor
};
/** Represents a learnable tensor */
struct Variable : public LayerGraph {
		static int variable_no;
		FGraphNode *node = nullptr;
		Variable() : LayerGraph(1) {
			name = "Variable" + std::to_string(variable_no++);
		}
		Variable(FGraphNode *node) : LayerGraph(1), node(node) {
			node->reference_counter++;
			name = "Variable" + std::to_string(variable_no++);
		}
		~Variable() {
			if (node) {
				node->reference_counter--;
			}
		}
		void forward() override { output[0] = node; }
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			// already included as an initializer
		}
		std::vector<std::vector<size_t>> propagate_shape(
			const std::vector<std::vector<size_t>> &input) override {
			return {std::vector<size_t>(node->operation.shape,
										node->operation.shape +
											node->operation.dimensions)};
		}
		static Variable *fromConstant(std::vector<size_t> shape,
									  float constant) {
			return new Variable(
				fconstant_f(constant, shape.data(), shape.size()));
		}
		static Variable *fromUniformRandom(std::vector<size_t> shape,
										   float minval = -0.15,
										   float maxval = 0.15) {
			FGraphNode *node = frandom(shape.data(), shape.size());
			FGraphNode *scaled =
				fsub_cf(fmul_cf(node, (maxval - minval)), minval);
			return new Variable(scaled);
		}
		static Variable *fromGlorotUniform(std::vector<size_t> shape) {
			unsigned int fan_in, fan_out;
			compute_fans(shape, fan_in, fan_out);
			double limit = std::sqrt(6. / (fan_in + fan_out));
			FGraphNode *node1 =
				fmax(frandom(shape.data(), (unsigned int)shape.size()),
					 std::numeric_limits<double>::epsilon());
			FGraphNode *node2 =
				fmax(frandom(shape.data(), (unsigned int)shape.size()),
					 std::numeric_limits<double>::epsilon());
			float sigma = std::sqrt(6. / (fan_in + fan_out));
			float mu = 0.0;
			FGraphNode *res =
				fadd_cf(fmul(fmul_cf(fsqrt_g(fmul_cf(flog(node1), -2)), sigma),
							 fcos(fmul_cf(node2, 2 * M_PI))),
						mu);
			return new Variable(res);
		}
};
/** Represents a constant tensor */
struct ConstantNode : public LayerGraph {
		FGraphNode *node;
		ConstantNode(float data, std::vector<size_t> shape) : LayerGraph(1) {
			node = fconstant_f(data, shape.data(), shape.size());
			name = "constant: " + std::to_string(data);
		}
		~ConstantNode() {
			if (node) {
				fFreeGraph(node);
			}
		}
		void forward() override {
			output.clear();
			output.push_back(node);
		}
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			// only referenced by its name
		}
};
/** Input Layers */
struct InputNode : public LayerGraph {
		static int data_no;
		std::vector<FGraphNode *> nodes;
		bool transpose_channel = false;
		InputNode() : LayerGraph(1) {
			int no = data_no++;
			name = "data";
			if (no)
				name += std::to_string(no);
		}
		InputNode(int no_nodes) : LayerGraph(no_nodes) {
			int no = data_no++;
			name = "data";
			if (no)
				name += std::to_string(no);
		}
		void forward() override {
			output.clear();
			for (FGraphNode *node : nodes)
				output.push_back(node);
		}
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			// only referenced by its name
		}
};
/** Relu activation layer */
struct Relu : public LayerGraph {
		static int relu_no;
		Relu() : LayerGraph(1) { name = "Relu" + std::to_string(relu_no++); }
		Relu(LayerGraph *in) : LayerGraph(in) {
			name = "Relu" + std::to_string(relu_no++);
		}
		void forward() override;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("Relu");
		}
};
/** Dropout layer (randomly sets entries of its input to 0 to avoid
 * overfitting). The dropout probability is given by a second input (expects a
 * 1D constant float). */
struct Dropout : public LayerGraph {
		static int drop_no;
		Dropout() : LayerGraph(1) {
			name = "Dropout" + std::to_string(drop_no++);
		}
		Dropout(LayerGraph *in) : LayerGraph(in) {
			name = "Dropout" + std::to_string(drop_no++);
		}
		void forward() override;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("Dropout");
		}
};
/** Softmax classification layer */
struct Softmax : public LayerGraph {
		int axis = 0;
		static int softmax_no;
		/** Initializes the SoftMax function with an optional
		 * axis parameter that describes the dimension of which
		 * the sum will be taken (may be negative in which case
		 * it will index from back, i.e. -1 means the last axis,
		 * -2 the one before the last etc.). Calculates `exp(in)
		 * / sum(in, ax)` */
		Softmax(int axis = -1) : LayerGraph(1), axis(axis) {
			name = "Softmax" + std::to_string(softmax_no++);
		}
		Softmax(LayerGraph *in, int axis = -1) : LayerGraph(in), axis(axis) {
			name = "Softmax" + std::to_string(softmax_no++);
		}
		void forward() override;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("Softmax");
			auto attr_ax = node->add_attribute();
			attr_ax->set_name("axis");
			attr_ax->set_i(axis);
		}
};
/** Flattens a dimension of its input */
struct Flatten : public LayerGraph {
		static int flatten_no;
		Flatten() : LayerGraph(1) {
			name = "Flatten" + std::to_string(flatten_no++);
		}
		Flatten(LayerGraph *in) : LayerGraph(in) {
			name = "Flatten" + std::to_string(flatten_no++);
		}
		void forward() override;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("Flatten");
		}
		virtual std::vector<std::vector<size_t>> propagate_shape(
			const std::vector<std::vector<size_t>> &input) override {
			std::vector<size_t> in = input[0];
			size_t total = 1;
			for (int i = 1; i < in.size(); i++)
				total *= in[i];
			return {{in[0], total}};
		}
};
/** Sums to inputs */
struct Add : public LayerGraph {
		static int add_no;
		Add() : LayerGraph(1) { name = "Add" + std::to_string(add_no++); }
		Add(LayerGraph *a, LayerGraph *b) : LayerGraph({a, b}) {
			name = "Add" + std::to_string(add_no++);
		}
		void forward() override;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("Add");
		}
};
static void deserialize_stride_and_padding(onnx::NodeProto *node,
										   std::vector<unsigned int> &stride,
										   std::vector<unsigned int> &padding) {
	auto attr = node->add_attribute();
	attr->set_name("pads");
	for (unsigned int p : padding)
		attr->add_ints(p);
	attr = node->add_attribute();
	attr->set_name("strides");
	for (unsigned int s : stride)
		attr->add_ints(s);
}
inline std::vector<size_t>
convolve_shape_transform(const std::vector<size_t> in,
						 const std::vector<size_t> filter,
						 const std::vector<unsigned int> &padding,
						 const std::vector<unsigned int> &stride) {
	using namespace std;
	vector<size_t> out_shape(in.size());
	out_shape[0] = in[0];	  // batch size
	out_shape[1] = filter[0]; // filters
	const size_t spatial_dims = in.size() - 2;
	auto padding_for = [&](size_t spatial_idx,
						   bool end_padding) -> unsigned int {
		if (padding.empty())
			return 0;
		if (padding.size() >= spatial_dims * 2)
			return padding[spatial_idx + (end_padding ? spatial_dims : 0)];
		if (padding.size() >= spatial_dims)
			return padding[spatial_idx];
		flogging(F_ERROR, "Invalid padding size for convolution layer.");
		return 0;
	};
	for (int i = 2; i < in.size(); i++) {
		size_t padded_shape = in[i];
		if (padding.size() != 0) {
			padded_shape +=
				padding_for(i - 2, false) + padding_for(i - 2, true);
		}
		std::cout << "padding " << i << ":" << padded_shape << ", ";
		size_t window_size = padded_shape - filter[i] + 1;
		std::cout << "window_size: " << padded_shape << " - " << filter[i]
				  << ", stride: " << stride[i - 2];
		window_size = window_size % stride[i - 2] == 0
						  ? window_size / stride[i - 2]
						  : window_size / stride[i - 2] + 1;
		std::cout << ", final: " << window_size << std::endl;
		out_shape[i] = window_size;
	}
	return out_shape;
}
/** Convolution layer.
 * Slides a set of filter kernels along the input and optionally adds a bias.
 * The input has `[batch, channels, h, w, ...]`, the kernel weights have
 * `[filters, channels, p, q, ...]`, the bias `[filters]` and the output has
 * `[batch, filters, y, x, ...]` where `y` and `x` depend on `stride` and
 * `padding`. `stride` defines the step size of the kernel window and `padding`
 * adds a zero border around the spatial dimensions to control the output size
 * and include the border elements.
 */
struct Convolve : public LayerGraph {
		static int conv_no;
		std::vector<unsigned int> stride, padding;
		Convolve() : LayerGraph(1) {
			name = "Conv" + std::to_string(conv_no++);
		}
		Convolve(std::vector<unsigned int> stride,
				 std::vector<unsigned int> padding)
			: LayerGraph(1), stride(stride), padding(padding) {
			name = "Conv" + std::to_string(conv_no++);
		}
		Convolve(std::vector<unsigned int> stride,
				 std::vector<unsigned int> padding, LayerGraph *in,
				 Variable *kernel, Variable *bias)
			: LayerGraph({in, kernel, bias}), stride(stride), padding(padding) {
			name = "Conv" + std::to_string(conv_no++);
		}
		Convolve(std::vector<unsigned int> stride,
				 std::vector<unsigned int> padding, Variable *kernel,
				 Variable *bias)
			: LayerGraph({kernel, bias}), stride(stride), padding(padding) {
			name = "Conv" + std::to_string(conv_no++);
		}
		void forward() override;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("Conv");
			deserialize_stride_and_padding(node, stride, padding);
		}
		std::vector<std::vector<size_t>> propagate_shape(
			const std::vector<std::vector<size_t>> &input) override {
			using namespace std;
			return {
				convolve_shape_transform(input[0], input[1], padding, stride)};
		}
};
static void deserialize_kernel_shape(onnx::NodeProto *node,
									 std::vector<size_t> &kernel) {
	auto attr = node->add_attribute();
	attr->set_name("kernel_shape");
	for (size_t s : kernel)
		attr->add_ints(s);
}
inline std::vector<size_t>
pooling_shape_transform(const std::vector<size_t> in,
						const std::vector<size_t> filter,
						const std::vector<unsigned int> &padding,
						const std::vector<unsigned int> &stride) {
	using namespace std;
	vector<size_t> out_shape(in.size());
	out_shape[0] = in[0]; // batch size
	out_shape[1] = in[1];
	const size_t spatial_dims = in.size() - 2;
	auto padding_for = [&](size_t spatial_idx,
						   bool end_padding) -> unsigned int {
		if (padding.empty())
			return 0;
		if (padding.size() >= spatial_dims * 2)
			return padding[spatial_idx + (end_padding ? spatial_dims : 0)];
		if (padding.size() >= spatial_dims)
			return padding[spatial_idx];
		flogging(F_ERROR, "Invalid padding size for pooling layer.");
		return 0;
	};
	for (int i = 2; i < in.size(); i++) {
		size_t padded_shape = in[i];
		if (padding.size() != 0) {
			padded_shape +=
				padding_for(i - 2, false) + padding_for(i - 2, true);
		}
		size_t window_size = padded_shape - filter[i - 2] + 1;
		window_size = window_size % stride[i - 2] == 0
						  ? window_size / stride[i - 2]
						  : window_size / stride[i - 2] + 1;
		out_shape[i] = window_size;
	}
	return out_shape;
}
/** Max pooling layer.
 * Reduces each spatial window to its maximum value. The window size is given by
 * `kernel_shape`, the step size by `stride`. Padding works like for
 * convolution: it adds a zero border around the spatial dimensions before
 * pooling. Batch and channel dimensions are preserved.
 */
struct MaxPool : public LayerGraph {
		static int mpool_no;
		std::vector<size_t> kernel_shape;
		std::vector<unsigned int> stride, padding;
		MaxPool() : LayerGraph(1) {
			name = "MaxPool" + std::to_string(mpool_no++);
		}
		MaxPool(std::vector<size_t> kernel_shape,
				std::vector<unsigned int> stride,
				std::vector<unsigned int> padding)
			: LayerGraph(1), kernel_shape(kernel_shape), stride(stride),
			  padding(padding) {
			name = "MaxPool" + std::to_string(mpool_no++);
		}
		MaxPool(std::vector<size_t> kernel_shape,
				std::vector<unsigned int> stride,
				std::vector<unsigned int> padding, LayerGraph *in)
			: LayerGraph(in), kernel_shape(kernel_shape), stride(stride),
			  padding(padding) {
			name = "MaxPool" + std::to_string(mpool_no++);
		}
		void forward() override;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("MaxPool");
			deserialize_stride_and_padding(node, stride, padding);
			deserialize_kernel_shape(node, kernel_shape);
		}
		std::vector<std::vector<size_t>> propagate_shape(
			const std::vector<std::vector<size_t>> &input) override {
			using namespace std;
			return {pooling_shape_transform(input[0], kernel_shape, padding,
											stride)};
		}
};
/** Average pooling layer.
 * Reduces each spatial window to its average value. The window size is given by
 * `kernel_shape`, the step size by `stride`, and padding behaves like in
 * convolution. Batch and channel dimensions are preserved.
 */
struct AvgPool : public LayerGraph {
		static int apool_no;
		std::vector<size_t> kernel_shape;
		std::vector<unsigned int> stride, padding;
		AvgPool() : LayerGraph(1) {
			name = "AvgPool" + std::to_string(apool_no++);
		}
		AvgPool(std::vector<size_t> kernel_shape,
				std::vector<unsigned int> stride,
				std::vector<unsigned int> padding)
			: LayerGraph(1), kernel_shape(kernel_shape), stride(stride),
			  padding(padding) {
			name = "AvgPool" + std::to_string(apool_no++);
		}
		AvgPool(std::vector<size_t> kernel_shape,
				std::vector<unsigned int> stride,
				std::vector<unsigned int> padding, LayerGraph *in)
			: LayerGraph(in), kernel_shape(kernel_shape), stride(stride),
			  padding(padding) {
			name = "AvgPool" + std::to_string(apool_no++);
		}
		void forward() override;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("AvgPool");
			deserialize_stride_and_padding(node, stride, padding);
			deserialize_kernel_shape(node, kernel_shape);
		}
		std::vector<std::vector<size_t>> propagate_shape(
			const std::vector<std::vector<size_t>> &input) override {
			using namespace std;
			return {pooling_shape_transform(input[0], kernel_shape, padding,
											stride)};
		}
};
/** Global average pooling layer.
 * Averages over all spatial dimensions so each channel is reduced to a single
 * value per batch. The output keeps the batch and channel dimensions and sets
 * all spatial dimensions to 1.
 */
struct GlobalAvgPool : public LayerGraph {
		static int gapool_no;
		GlobalAvgPool() : LayerGraph(1) {
			name = "GlobalAvgPool" + std::to_string(gapool_no++);
		}
		GlobalAvgPool(LayerGraph *in) : LayerGraph(in) {
			name = "GlobalAvgPool" + std::to_string(gapool_no++);
		}
		void forward() override;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("GlobalAveragePool");
		}
		std::vector<std::vector<size_t>> propagate_shape(
			const std::vector<std::vector<size_t>> &input) override {
			using namespace std;
			const int dims = input[0].size();
			vector<size_t> rank_shape(dims, 1);
			rank_shape[0] = input[0][0];
			rank_shape[1] = input[0][1];
			return {rank_shape};
		}
};
/** Batch normalization layer.
 * Normalizes the input per channel and applies a learnable scale (gamma) and
 * bias (beta). If running mean and variance are provided they are updated with
 * momentum `alpha` when training, otherwise they are computed on the fly.
 */
struct BatchNorm : public LayerGraph {
		// TODO dynamically determine if running mean and variance should be
		// used as additional outputs
		static int bnorm_no;
		BatchNorm(float alpha = 0.8) : LayerGraph(1), alpha(alpha) {
			name = "BatchNorm" + std::to_string(bnorm_no++);
		}
		BatchNorm(LayerGraph *in, Variable *gamma, Variable *beta,
				  float alpha = 0.8)
			: LayerGraph({in, gamma, beta}), alpha(alpha) {
			name = "BatchNorm" + std::to_string(bnorm_no++);
		}
		void forward() override;
		float alpha;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("BatchNormalization");
			auto attr_alpha = node->add_attribute();
			attr_alpha->set_name("momentum");
			attr_alpha->set_f(alpha);
		}
};
/** Fully connected (dense) layer. */
struct Connected : public LayerGraph {
		static int connected_no;
		Connected() : LayerGraph(1) {
			name = "Connected" + std::to_string(connected_no++);
		}
		Connected(LayerGraph *in, Variable *weight, Variable *bias = nullptr,
				  bool transposeA = false, bool transposeB = false)
			: LayerGraph({in, weight, bias}), transposeA(transposeA),
			  transposeB(transposeB) {
			name = "Connected" + std::to_string(connected_no++);
		}
		Connected(Variable *weight, Variable *bias = nullptr,
				  bool transposeA = false, bool transposeB = false)
			: LayerGraph({weight, bias}), transposeA(transposeA),
			  transposeB(transposeB) {
			name = "Connected" + std::to_string(connected_no++);
		}
		void forward() override;
		bool transposeA = false;
		bool transposeB = false;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("Gemm");
			auto attr_a = node->add_attribute();
			attr_a->set_name("transA");
			attr_a->set_i(transposeA ? 1 : 0);
			auto attr_b = node->add_attribute();
			attr_b->set_name("transB");
			attr_b->set_i(transposeB ? 1 : 0);
		}
		std::vector<std::vector<size_t>> propagate_shape(
			const std::vector<std::vector<size_t>> &input) override {
			using namespace std;
			const vector<size_t> &img = input[0];
			const vector<size_t> &filter = input[1];
			vector<size_t> output(img);
			size_t n = transposeA ? output[output.size() - 1]
								  : output[output.size() - 2];
			size_t k = transposeB ? filter[filter.size() - 2]
								  : filter[filter.size() - 1];
			output[output.size() - 2] = n;
			output[output.size() - 1] = k;
			return {output};
		}
};

/** High level layer factories and configuration helpers. */
namespace FlintDL {

/** Padding for convolution and pooling.
 * - `Valid` uses no padding (only fully covered windows are used)
 * - `Same` adds padding so the output keeps its spatial size (for odd kernels)
 * - `Explicit` uses the `padding` values from the options
 */
enum class PaddingMode { Valid, Same, Explicit };
/** Weight initializer kind for trainable weights. */
enum class InitializerKind { GlorotUniform, Uniform, Constant };
/** Activation layer kind for builder helpers. */
enum class ActivationKind { None, Relu, Softmax };

/** Options for a 2D convolution layer.
 * - `stride` step size of the kernel window
 * - `padding_mode` controls how padding is generated
 * - `padding` is used only if `padding_mode` is `Explicit`
 * - `use_bias` adds a trainable bias per filter
 * - `bias_init` initializes the bias if enabled
 * - `kernel_initializer` selects the weight initializer
 * - `uniform_min`/`uniform_max` are used for the uniform initializer
 * - `kernel_constant` is used for constant initialization
 */
struct Conv2DOptions {
		std::array<unsigned int, 2> stride{1, 1};
		PaddingMode padding_mode = PaddingMode::Valid;
		std::array<unsigned int, 2> padding{0, 0};
		bool use_bias = true;
		float bias_init = 0.0f;
		InitializerKind kernel_initializer = InitializerKind::GlorotUniform;
		float uniform_min = -0.15f;
		float uniform_max = 0.15f;
		float kernel_constant = 0.0f;
};

/** Options for a dense layer.
 * - `use_bias` adds a trainable bias
 * - `bias_init` initializes the bias if enabled
 * - `transpose_input` transposes the input before multiplication
 * - `transpose_kernel` transposes the kernel weights before multiplication
 * - `kernel_initializer` selects the weight initializer
 */
struct DenseOptions {
		bool use_bias = true;
		float bias_init = 0.0f;
		bool transpose_input = false;
		bool transpose_kernel = false;
		InitializerKind kernel_initializer = InitializerKind::GlorotUniform;
		float uniform_min = -0.15f;
		float uniform_max = 0.15f;
		float kernel_constant = 0.0f;
};

/** Options for pooling layers.
 * - `stride` step size of the pooling window (0 means kernel size)
 * - `padding_mode` controls how padding is generated
 * - `padding` is used only if `padding_mode` is `Explicit`
 */
struct Pool2DOptions {
		std::array<unsigned int, 2> stride{0, 0};
		PaddingMode padding_mode = PaddingMode::Valid;
		std::array<unsigned int, 2> padding{0, 0};
};

namespace detail {
inline Variable *make_variable(std::vector<size_t> shape,
							   InitializerKind init_kind, float uniform_min,
							   float uniform_max, float constant) {
	switch (init_kind) {
	case InitializerKind::GlorotUniform:
		return Variable::fromGlorotUniform(shape);
	case InitializerKind::Uniform:
		return Variable::fromUniformRandom(shape, uniform_min, uniform_max);
	case InitializerKind::Constant:
		return Variable::fromConstant(shape, constant);
	}
	throw std::runtime_error("Unknown initializer kind");
}

inline std::vector<unsigned int>
resolve_padding_2d(std::array<size_t, 2> kernel, PaddingMode mode,
				   std::array<unsigned int, 2> padding) {
	switch (mode) {
	case PaddingMode::Valid:
		return {0, 0};
	case PaddingMode::Same:
		return {static_cast<unsigned int>(kernel[0] / 2),
				static_cast<unsigned int>(kernel[1] / 2)};
	case PaddingMode::Explicit:
		return {padding[0], padding[1]};
	}
	throw std::runtime_error("Unknown padding mode");
}

inline std::vector<unsigned int>
resolve_stride_2d(std::array<unsigned int, 2> stride,
				  std::array<size_t, 2> kernel) {
	if (stride[0] == 0 || stride[1] == 0)
		return {static_cast<unsigned int>(kernel[0]),
				static_cast<unsigned int>(kernel[1])};
	return {stride[0], stride[1]};
}
} // namespace detail

/**
 * Constructs an Activation Layer from an `ActivationKind`
 * (None, Relu, or Softmax).
 */
inline LayerGraph *activation_layer(ActivationKind activation) {
	switch (activation) {
	case ActivationKind::None:
		return nullptr;
	case ActivationKind::Relu:
		return new Relu();
	case ActivationKind::Softmax:
		return new Softmax();
	}
	throw std::runtime_error("Unknown activation kind");
}

/**
 * Chains a newly constructed activation layer (from a `ActivationKind`) to its
 * input node (i.e. input -> activation) and returns the activation layer.
 */
inline LayerGraph *apply_activation(LayerGraph *in, ActivationKind activation) {
	LayerGraph *layer = activation_layer(activation);
	if (!layer)
		return in;
	layer->incoming.push_back(in);
	in->outgoing.push_back(layer);
	return layer;
}
/**
 * Constructs a 2D convolution layer and initializes its weights.
 * - `filters` number of filter kernels
 * - `kernel_size` shape of filter kernels
 * - `in_channels` number of channels of the input image (e.g., 3 for RGB)
 * - `options` control stride, padding, bias and initialization
 */
inline Convolve *conv2d(size_t filters, std::array<size_t, 2> kernel_size,
						size_t in_channels,
						const Conv2DOptions &options = Conv2DOptions()) {
	if (filters == 0 || kernel_size[0] == 0 || kernel_size[1] == 0 ||
		in_channels == 0)
		throw std::runtime_error("conv2d expects filters/channels/kernel");

	Variable *kernel = detail::make_variable(
		{filters, in_channels, kernel_size[0], kernel_size[1]},
		options.kernel_initializer, options.uniform_min, options.uniform_max,
		options.kernel_constant);
	Variable *bias = options.use_bias
						 ? Variable::fromConstant({filters}, options.bias_init)
						 : nullptr;
	return new Convolve(detail::resolve_stride_2d(options.stride, kernel_size),
						detail::resolve_padding_2d(
							kernel_size, options.padding_mode, options.padding),
						kernel, bias);
}

/**
 * Constructs a fully connected layer and initializes its weights.
 * - `in_units` number of input units
 * - `out_units` number of output units
 * - `options` control bias, transpositions and initialization
 */
inline Connected *dense(size_t in_units, size_t out_units,
						const DenseOptions &options = DenseOptions()) {
	if (in_units == 0 || out_units == 0)
		throw std::runtime_error("dense expects non-zero in/out units");
	Variable *kernel = detail::make_variable(
		{in_units, out_units}, options.kernel_initializer, options.uniform_min,
		options.uniform_max, options.kernel_constant);
	Variable *bias =
		options.use_bias
			? Variable::fromConstant({out_units}, options.bias_init)
			: nullptr;
	return new Connected(kernel, bias, options.transpose_input,
						 options.transpose_kernel);
}

/**
 * Constructs a max pooling layer.
 * - `kernel_size` pooling window size
 * - `options` control stride and padding
 */
inline MaxPool *maxpool2d(std::array<size_t, 2> kernel_size,
						  const Pool2DOptions &options = Pool2DOptions()) {
	if (kernel_size[0] == 0 || kernel_size[1] == 0)
		throw std::runtime_error(
			"maxpool2d expects non-zero kernel dimensions");
	return new MaxPool({kernel_size[0], kernel_size[1]},
					   detail::resolve_stride_2d(options.stride, kernel_size),
					   detail::resolve_padding_2d(
						   kernel_size, options.padding_mode, options.padding));
}

/**
 * Constructs a max pooling layer with explicit stride.
 */
inline MaxPool *maxpool2d(std::array<size_t, 2> kernel_size,
						  std::array<unsigned int, 2> stride) {
	Pool2DOptions options;
	options.stride = stride;
	return maxpool2d(kernel_size, options);
}

/**
 * Constructs an average pooling layer.
 * - `kernel_size` pooling window size
 * - `options` control stride and padding
 */
inline AvgPool *avgpool2d(std::array<size_t, 2> kernel_size,
						  const Pool2DOptions &options = Pool2DOptions()) {
	if (kernel_size[0] == 0 || kernel_size[1] == 0)
		throw std::runtime_error(
			"avgpool2d expects non-zero kernel dimensions");
	return new AvgPool({kernel_size[0], kernel_size[1]},
					   detail::resolve_stride_2d(options.stride, kernel_size),
					   detail::resolve_padding_2d(
						   kernel_size, options.padding_mode, options.padding));
}

/**
 * Constructs an average pooling layer with explicit stride.
 */
inline AvgPool *avgpool2d(std::array<size_t, 2> kernel_size,
						  std::array<unsigned int, 2> stride) {
	Pool2DOptions options;
	options.stride = stride;
	return avgpool2d(kernel_size, options);
}

/** Constructs a flatten layer. */
inline Flatten *flatten() { return new Flatten(); }
/** Constructs a ReLU activation layer. */
inline Relu *relu() { return new Relu(); }
/** Constructs a softmax activation layer. */
inline Softmax *softmax(int axis = -1) { return new Softmax(axis); }
/** Constructs a dropout layer with the given probability. */
inline Dropout *dropout(float probability) {
	if (probability < 0.0f || probability > 1.0f)
		throw std::runtime_error("dropout probability must be in [0, 1]");
	return new Dropout(new ConstantNode(probability, {1, 1}));
}

} // namespace FlintDL

#endif
