#include "../layers.hpp"
#include "flint.h"
int Relu::relu_no = 0;
void Relu::forward() {
#ifdef FLINT_DEBUG
	if (incoming.size() != 1)
		flogging(F_ERROR, "Relu expects exactly one input layer, not " +
							  std::to_string(incoming.size()));
	if (incoming[0]->output.size() != 1)
		flogging(F_ERROR,
				 "Relu expects exactly one input, previous layer gave " +
					 std::to_string(incoming[0]->output.size()));
#endif
	output[0] = fmax_ci(incoming[0]->output[0], 0);
}
int Softmax::softmax_no = 0;
void Softmax::forward() {
#ifdef FLINT_DEBUG
	if (incoming.size() != 1)
		flogging(F_ERROR, "Softmax expects exactly one input layer, not " +
							  std::to_string(incoming.size()));
	if (incoming[0]->output.size() != 1)
		flogging(F_ERROR,
				 "Softmax expects exactly one input, previous layer gave " +
					 std::to_string(incoming[0]->output.size()));
#endif
	FGraphNode *in = incoming[0]->output[0];
	const unsigned int n = in->operation.dimensions;
	const unsigned int ax = axis < 0 ? n + axis : axis;
	// numerical stability
	FGraphNode *exp = fexp(
		fsub(in, fexpand(freduce_max(in, ax), ax, in->operation.shape[ax])));
	FGraphNode *sum = freduce_sum(exp, ax);
	if (ax == 0 || n == 1) {
		output[0] = fdiv_g(exp, sum);
	} else {
		output[0] = fdiv_g(exp, fexpand(sum, ax, in->operation.shape[ax]));
	}
}
