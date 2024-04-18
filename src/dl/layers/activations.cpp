#include "flint.h"
#include "../layers.hpp"

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

