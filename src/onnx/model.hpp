#ifndef ONNX_MODEL
#define ONNX_MODEL
#include "flint.h"
#include "layers.hpp"
#include <list>
struct GraphModel {
		InputNode *input;
    std::vector<Variable *> weights;
		LayerGraph *output;
		GraphModel() = default;
		FGraphNode *operator()(FGraphNode *in);
		static GraphModel load_model(std::string path);
		// TODO training
};

#endif
