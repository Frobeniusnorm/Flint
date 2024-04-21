#ifndef ONNX_MODEL
#define ONNX_MODEL
#include "flint.h"
#include "layers.hpp"
struct GraphModel {
		// TODO multiple inputs
		InputNode *input;
		std::vector<Variable *> weights;
		LayerGraph *output;
		GraphModel() = default;
		FGraphNode *operator()(FGraphNode *in);
		std::vector<FGraphNode*> operator()(std::initializer_list<FGraphNode *> in);
		static GraphModel *load_model(std::string path);
		static GraphModel *sequential(std::initializer_list<LayerGraph *> list);
		// TODO training
};

#endif
