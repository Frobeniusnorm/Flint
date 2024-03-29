#include "flint.h"
#include "../layers.hpp"
#include <string>
#include <array>

void Flatten::forward() {
#ifdef FLINT_DEBUG
	if (incoming.size() != 1)
		flogging(F_ERROR, "Flatten expects exactly one input layer, not " +
							  std::to_string(incoming.size()));
	if (incoming[0]->output.size())
		flogging(F_ERROR,
				 "Flatten expects exactly one input, previous layer gave " +
					 std::to_string(incoming[0]->output.size()));
#endif
	// keep first dimension by reshaping
	using namespace std;
	FGraphNode* in = incoming[0]->output[0];
	array<size_t, 2> flattened_shape;
	flattened_shape[0] = in->operation.shape[0];
	flattened_shape[1] = 1;
	for (int i = 0; i < in->operation.dimensions; i++)
		flattened_shape[1] *= in->operation.shape[i];
	output[0] = freshape(in, flattened_shape.data(), 2);
}
