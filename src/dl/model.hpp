#ifndef ONNX_MODEL
#define ONNX_MODEL
#include "layers.hpp"
#include <flint/flint.h>
#include <functional>
#include <map>
#include <optional>

struct SequentialBuilder;
/**
 * A Model for neural networks that represent the connections between the layers
 * as an acyclic graph. This allows arbitrary topology of the model.
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
		 * counter should be incremented).
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
		 * incremented).
		 */
		std::vector<FGraphNode *> operator()(
			std::vector<FGraphNode *> in,
			std::optional<std::reference_wrapper<std::map<LayerGraph *, long>>>
				time_per_layer = std::nullopt);
		std::string serialize_onnx();
		std::vector<std::vector<size_t>>
		shape_interference(std::vector<std::vector<size_t>> input_shapes);
		static GraphModel *load_model(std::string path);
		static GraphModel *sequential(std::vector<LayerGraph *> list);
		static GraphModel *from_output(LayerGraph *output);
		static SequentialBuilder builder();
};

struct SequentialBuilder {
		std::vector<LayerGraph *> layers;

		SequentialBuilder &add(LayerGraph *layer) {
			layers.push_back(layer);
			return *this;
		}

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

		SequentialBuilder &maxpool2d(
			std::array<size_t, 2> kernel_size,
			const FlintDL::Pool2DOptions &options = FlintDL::Pool2DOptions()) {
			return add(FlintDL::maxpool2d(kernel_size, options));
		}

		SequentialBuilder &maxpool2d(std::array<size_t, 2> kernel_size,
									 std::array<unsigned int, 2> stride) {
			return add(FlintDL::maxpool2d(kernel_size, stride));
		}

		SequentialBuilder &avgpool2d(
			std::array<size_t, 2> kernel_size,
			const FlintDL::Pool2DOptions &options = FlintDL::Pool2DOptions()) {
			return add(FlintDL::avgpool2d(kernel_size, options));
		}

		SequentialBuilder &avgpool2d(std::array<size_t, 2> kernel_size,
									 std::array<unsigned int, 2> stride) {
			return add(FlintDL::avgpool2d(kernel_size, stride));
		}

		SequentialBuilder &flatten() { return add(FlintDL::flatten()); }
		SequentialBuilder &relu() { return add(FlintDL::relu()); }
		SequentialBuilder &softmax(int axis = -1) {
			return add(FlintDL::softmax(axis));
		}
		SequentialBuilder &dropout(float probability) {
			return add(FlintDL::dropout(probability));
		}

		GraphModel *build() { return GraphModel::sequential(layers); }
};

inline SequentialBuilder GraphModel::builder() { return SequentialBuilder(); }

#endif
