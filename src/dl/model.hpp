#ifndef ONNX_MODEL
#define ONNX_MODEL
#include "layers.hpp"
#include <flint/flint.h>
#include <functional>
#include <map>
#include <optional>

struct SequentialBuilder;
/**
 * A Model for neural networks that represent the connections between the
 * layers as an acyclic graph. This allows arbitrary topology of the model.
 * In- and Export is implemented for the ONNX specification.
 */
struct GraphModel {
		std::vector<InputNode *> input;
		std::vector<Variable *> weights;
		std::vector<LayerGraph *> output;
		GraphModel() = default;
		/**
		 * Feeds the single input tensor through the model and returns a single
		 * output. This function should be used if you have a model with exactly
		 * one input and one output tensor. The input node is only preserved if
		 * its reference counter is >= 1. The output node has a reference
		 * counter of 0 (so if it is used in further calculations, its reference
		 * counter should be incremented). If `time_per_layer` is provided, it
		 * is filled with per layer execution time in nanoseconds.
		 */
		FGraphNode *operator()(
			FGraphNode *in,
			std::optional<std::reference_wrapper<std::map<LayerGraph *, long>>>
				time_per_layer = std::nullopt);
		/**
		 * Feeds all input tensors through the model and returns the output
		 * tensors. The input nodes are only preserved if its reference counter
		 * is >= 1. The output node has a reference counter of 0 (so if it is
		 * used in further calculations, its reference counter should be
		 * incremented). If `time_per_layer` is provided, it is filled with per
		 * layer execution time in nanoseconds.
		 */
		std::vector<FGraphNode *> operator()(
			std::vector<FGraphNode *> in,
			std::optional<std::reference_wrapper<std::map<LayerGraph *, long>>>
				time_per_layer = std::nullopt);
		/** Serializes the model into an ONNX string. */
		std::string serialize_onnx();
		/**
		 * Infers all output shapes for the given input shapes. The returned
		 * shapes are ordered like the output nodes of the model.
		 */
		std::vector<std::vector<size_t>>
		shape_interference(std::vector<std::vector<size_t>> input_shapes);
		/** Loads an ONNX model from disk. */
		static GraphModel *load_model(std::string path);
		/** Builds a sequential model from the given list of layers. */
		static GraphModel *sequential(std::vector<LayerGraph *> list);
		/** Builds a model by tracing all incoming layers of one output node. */
		static GraphModel *from_output(LayerGraph *output);
		/** Returns a SequentialBuilder helper for fluent model construction. */
		static SequentialBuilder builder();
};
/**
 * For building a sequential (layer following layer) GraphModel
 */
struct SequentialBuilder {
		std::vector<LayerGraph *> layers;
		/** Adds an arbitrary layer next */
		SequentialBuilder &add(LayerGraph *layer) {
			layers.push_back(layer);
			return *this;
		}
		/**
		 * Constructs and adds a 2D Convolution Layer with a default
		 * Conv2DOptions struct.
		 * - `filters` gives the number of filters to use (or "number of
		 * kernels")
		 * - `kernel_size` gives the shape of the individual filter kernels (2D)
		 * - `in_channels` gives the number of channels in the input tensor
		 * - `activation` adds an activation layer after the convolution
		 */
		SequentialBuilder &conv2d(size_t filters,
								  std::array<size_t, 2> kernel_size,
								  size_t in_channels,
								  FlintDL::ActivationKind activation =
									  FlintDL::ActivationKind::None) {
			return conv2d(filters, kernel_size, in_channels,
						  FlintDL::Conv2DOptions(), activation);
		}

		SequentialBuilder &conv2d(size_t filters,
								  std::array<size_t, 2> kernel_size,
								  size_t in_channels,
								  const FlintDL::Conv2DOptions &options,
								  FlintDL::ActivationKind activation =
									  FlintDL::ActivationKind::None) {
			LayerGraph *base =
				FlintDL::conv2d(filters, kernel_size, in_channels, options);
			add(base);
			LayerGraph *activated = FlintDL::activation_layer(activation);
			if (activated)
				add(activated);
			return *this;
		}

		/**
		 * Constructs and adds a fully connected layer with a default
		 * DenseOptions struct.
		 */
		SequentialBuilder &dense(size_t in_units, size_t out_units,
								 FlintDL::ActivationKind activation =
									 FlintDL::ActivationKind::None) {
			return dense(in_units, out_units, FlintDL::DenseOptions(),
						 activation);
		}

		SequentialBuilder &dense(size_t in_units, size_t out_units,
								 const FlintDL::DenseOptions &options,
								 FlintDL::ActivationKind activation =
									 FlintDL::ActivationKind::None) {
			LayerGraph *base = FlintDL::dense(in_units, out_units, options);
			add(base);
			LayerGraph *activated = FlintDL::activation_layer(activation);
			if (activated)
				add(activated);
			return *this;
		}

		/** Adds a max pooling layer. */
		SequentialBuilder &maxpool2d(
			std::array<size_t, 2> kernel_size,
			const FlintDL::Pool2DOptions &options = FlintDL::Pool2DOptions()) {
			return add(FlintDL::maxpool2d(kernel_size, options));
		}

		SequentialBuilder &maxpool2d(std::array<size_t, 2> kernel_size,
									 std::array<unsigned int, 2> stride) {
			return add(FlintDL::maxpool2d(kernel_size, stride));
		}

		/** Adds an average pooling layer. */
		SequentialBuilder &avgpool2d(
			std::array<size_t, 2> kernel_size,
			const FlintDL::Pool2DOptions &options = FlintDL::Pool2DOptions()) {
			return add(FlintDL::avgpool2d(kernel_size, options));
		}

		SequentialBuilder &avgpool2d(std::array<size_t, 2> kernel_size,
									 std::array<unsigned int, 2> stride) {
			return add(FlintDL::avgpool2d(kernel_size, stride));
		}

		/** Adds a flatten layer. */
		SequentialBuilder &flatten() { return add(FlintDL::flatten()); }
		/** Adds a ReLU activation layer. */
		SequentialBuilder &relu() { return add(FlintDL::relu()); }
		/** Adds a softmax activation layer. */
		SequentialBuilder &softmax(int axis = -1) {
			return add(FlintDL::softmax(axis));
		}
		/** Adds a dropout layer. */
		SequentialBuilder &dropout(float probability) {
			return add(FlintDL::dropout(probability));
		}

		/** Builds the sequential GraphModel. */
		GraphModel *build() { return GraphModel::sequential(layers); }
};

inline SequentialBuilder GraphModel::builder() { return SequentialBuilder(); }

#endif
