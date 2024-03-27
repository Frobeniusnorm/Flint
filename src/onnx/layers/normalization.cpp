#include "flint.h"
#include "layers.hpp"
#include <limits>

void BatchNorm::forward() {
#ifdef FLINT_DEBUG
	if (incoming.size() != 3 || incoming[0]->output.size() != 1 ||
		incoming[1]->output.size() != 1 || incoming[2]->output.size() != 1)
		flogging(F_ERROR, "BatchNorm expects an image and a gamma and beta "
						  "parameter as inputs");
#endif
	FGraphNode *x = incoming[0]->output[0];
	FGraphNode *gamma = incoming[1]->output[0];
	FGraphNode *beta = incoming[2]->output[0];
	// calculate mean
	FGraphNode *mean =
		fdiv_ci(freduce_sum(x, 0), (unsigned int)x->operation.shape[0]);
	FGraphNode *var = fsub_g(x, mean);
	if (!mean_running)
		mean_running = mean;
	else
		mean_running =
			fadd_g(fmul_cf(mean_running, alpha), fmul_cf(mean, 1 - alpha));
	if (!var_running)
		var_running = var;
	else
		var_running =
			fadd_g(fmul_cf(var_running, alpha), fmul_cf(var, 1 - alpha));
	FGraphNode *y = fadd_g(
		fmul_g(gamma,
			   fdiv_g(fsub_g(x, mean_running),
					  fsqrt_g(fadd_cf(fpow(var_running, 2),
									  std::numeric_limits<float>::epsilon())))),
		beta);
};
