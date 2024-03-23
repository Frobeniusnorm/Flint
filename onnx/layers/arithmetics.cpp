#include "../../flint.h"
#include "layers.hpp"
#include <string>

void Add::forward() {
#ifdef FLINT_DEBUG
	if (!((incoming.size() == 1 && incoming[0]->output.size() == 2) &&
		  (incoming.size() == 2 && incoming[0]->output.size() == 1 &&
		   incoming[1]->output.size() == 1)))
		flogging(F_ERROR,
				 "Add expects exactly two inputs (either two layers with one "
				 "output each or one layer with two outputs)");
#endif
	if (incoming.size() == 2)
		output[0] = fadd(incoming[0]->output[0], incoming[1]->output[0]);
	else
		output[0] = fadd(incoming[0]->output[0], incoming[0]->output[1]);
}

void Connected::forward() {
#ifdef FLINT_DEBUG
	if (incoming.size() != 2 && incoming[0]->output.size() == 1 &&
		incoming[1]->output.size() == 1)
		flogging(
			F_ERROR,
			"Connected expects exactly two inputs, the input and the kernel");
#endif
	output[0] = fmatmul(incoming[0]->output[0], incoming[1]->output[0]);
}
