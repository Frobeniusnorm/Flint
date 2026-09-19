#include "../layers.hpp"
#include "flint.h"
#include <array>
#include <string>
int Flatten::flatten_no = 0;
void Flatten::forward() {
#ifdef FLINT_DEBUG
	if (incoming.size() != 1)
		flogging(F_ERROR, "Flatten expects exactly one input layer, not " +
							  std::to_string(incoming.size()));
	if (incoming[0]->output.size() != 1)
		flogging(F_ERROR,
				 "Flatten expects exactly one input, previous layer gave " +
					 std::to_string(incoming[0]->output.size()));
#endif
	// keep first dimension by reshaping
	using namespace std;
	FGraphNode *in = incoming[0]->output[0];
	// flatten channel major like ONNX does, so that the weights of a following
	// dense layer mean the same in both layouts. Without spatial extent both
	// orders are the same.
	const int dims = in->operation.dimensions;
	size_t spatial = 1;
	for (int i = 1; i < dims - 1; i++)
		spatial *= in->operation.shape[i];
	if (dims > 2 && spatial > 1) {
		vector<int> channels_first(dims);
		channels_first[0] = 0;
		channels_first[1] = dims - 1;
		for (int i = 2; i < dims; i++)
			channels_first[i] = i - 1;
		in = ftranspose(in, channels_first.data());
	}
	array<size_t, 2> flattened_shape;
	flattened_shape[0] = in->operation.shape[0];
	flattened_shape[1] = 1;
	for (int i = 1; i < in->operation.dimensions; i++)
		flattened_shape[1] *= in->operation.shape[i];
	output[0] = freshape(in, flattened_shape.data(), 2);
}
