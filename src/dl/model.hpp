#ifndef ONNX_MODEL
#define ONNX_MODEL
#include "flint.h"
#include "layers.hpp"
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
		FGraphNode *operator()(FGraphNode *in);
		std::vector<FGraphNode *> operator()(std::vector<FGraphNode *> in);
		std::string serialize();
		static GraphModel *load_model(std::string path);
		static GraphModel *sequential(std::vector<LayerGraph *> list);
		static GraphModel *from_output(LayerGraph *output);
		// TODO training
};

#endif
