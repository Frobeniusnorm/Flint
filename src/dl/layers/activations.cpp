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
	if (n == 2 && ax == 1) {
		const size_t channels = in->operation.shape[1];
		const size_t one_col_shape[] = {channels, 1};
		const size_t one_row_shape[] = {1, channels};
		FGraphNode *ones_col = in->operation.data_type == F_FLOAT32
								   ? fconstant_f(1.0f, one_col_shape, 2)
								   : fconstant_d(1.0, one_col_shape, 2);
		FGraphNode *ones_row = in->operation.data_type == F_FLOAT32
								   ? fconstant_f(1.0f, one_row_shape, 2)
								   : fconstant_d(1.0, one_row_shape, 2);
		FGraphNode *sum = fmatmul(exp, ones_col);
		FGraphNode *den = fmatmul(sum, ones_row);
		output[0] = fdiv_g(exp, den);
		return;
	}
	FGraphNode *sum = freduce_sum(exp, ax);
	if (ax == 0 || n == 1)
		output[0] = fdiv_g(exp, sum);
	else
		output[0] = fdiv_g(exp, fexpand(sum, ax, in->operation.shape[ax]));
}
