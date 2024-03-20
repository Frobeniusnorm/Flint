#include "../../flint.h"
#include "layers.hpp"
#include <string>

void Convolve::forward() {
#ifdef FLINT_DEBUG
	if ((incoming.size() != 2 && incoming.size() != 3) ||
		incoming[0]->output.size() != 1 || incoming[1]->output.size() != 1 ||
		(incoming.size() == 3 && incoming[2]->output.size() != 1))
		flogging(F_ERROR, "Convolve expects an image and a kernel as "
						  "parameters and optionally a bias");
#endif
	FGraphNode *weight = incoming[1]->output[0];
	FGraphNode *bias = incoming.size() == 3 ? incoming[2]->output[0] : nullptr;
	// expand kernel s.t. it matches the batch size
	FGraphNode *eweight = fexpand(weight, 0, 1);
	using namespace std;
	vector<unsigned int> steps(stride.size() + 1);
	steps[0] = 1;
	for (int i = 1; i < steps.size(); i++)
		steps[i] = stride[i];
	output[0] = fconvolve(incoming[0]->output[0], eweight, steps.data());
	if (bias)
		output[0] = fadd(output[0], bias);
}
