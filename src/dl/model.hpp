#ifndef ONNX_MODEL
#define ONNX_MODEL
#include "flint.h"
#include "layers.hpp"
struct GraphModel {
		// TODO multiple inputs
		std::vector<InputNode *> input;
		std::vector<Variable *> weights;
		std::vector<LayerGraph *> output;
		GraphModel() = default;
		FGraphNode *operator()(FGraphNode *in);
		std::vector<FGraphNode *> operator()(std::vector<FGraphNode *> in);
		static GraphModel *load_model(std::string path);
		static GraphModel *sequential(std::vector<LayerGraph *> list);
		static GraphModel *from_output(LayerGraph *output);
		// TODO training
};

#endif
