#ifndef ONNX_LAYERS_RELU
#define ONNX_LAYERS_RELU
#include "flint.h"
#include "layers.hpp"
#include <string>

void Relu::forward() {
#ifdef FLINT_DEBUG
	if (incoming.size() != 1)
		flogging(F_ERROR, "Relu expects exactly one input layer, not " +
							  std::to_string(incoming.size()));
	if (incoming[0]->output.size())
		flogging(F_ERROR,
				 "Relu expects exactly one input, previous layer gave " +
					 std::to_string(incoming[0]->output.size()));
#endif
	if (output[0])
		fFreeGraph(output[0]);
	output[0] = fmax_ci(incoming[0]->output[0], 0);
}

#endif
