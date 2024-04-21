#include "../layers.hpp"
#include "flint.h"
#include <limits>

void BatchNorm::forward() {
#ifdef FLINT_DEBUG
	if (incoming.size() != 5 || incoming[0]->output.size() != 1 ||
		incoming[1]->output.size() != 1 || incoming[2]->output.size() != 1)
		flogging(F_ERROR, "BatchNorm expects an image, a gamma, a beta, the "
						  "running mean and the running variance"
						  "parameter as inputs");
#endif
	FGraphNode *x = incoming[0]->output[0];
	FGraphNode *gamma = incoming[1]->output[0];
	FGraphNode *beta = incoming[2]->output[0];
	// because channels and other dimensions have to be swapped
	FGraphNode *mean_running =
		incoming.size() <= 3 ? nullptr : incoming[3]->output[0];
	FGraphNode *var_running =
		incoming.size() <= 3 ? nullptr : incoming[4]->output[0];
	int transpositions1[x->operation.dimensions];
	for (int i = 0; i < x->operation.dimensions; i++)
		transpositions1[i] = i;
	transpositions1[1] = x->operation.dimensions - 1;
	transpositions1[x->operation.dimensions - 1] = 1;
	FGraphNode *tx = ftranspose(x, transpositions1);
	// calculate mean and std var
	if (training && mean_running) {
		// mean and var for all except for the channels
		FGraphNode *mean = tx;
		while (mean->operation.dimensions > 1)
			mean = fdiv_ci(freduce_sum(mean, 0),
						   (unsigned int)mean->operation.shape[0]);
		FGraphNode *var = fsub_g(tx, mean);
		while (var->operation.dimensions > 1)
			var = fdiv_ci(freduce_sum(var, 0),
						  (unsigned int)var->operation.shape[0]);
		mean_running =
			fadd_g(fmul_cf(mean_running, alpha), fmul_cf(mean, 1 - alpha));
		var_running =
			fadd_g(fmul_cf(var_running, alpha), fmul_cf(var, 1 - alpha));
		Variable *node_mean = dynamic_cast<Variable *>(incoming[3]);
		Variable *node_var = dynamic_cast<Variable *>(incoming[4]);
		fExecuteGraph(mean_running);
		fExecuteGraph(var_running);
		node_mean->node->reference_counter--;
		node_var->node->reference_counter--;
		fFreeGraph(node_mean->node);
		fFreeGraph(node_var->node);
		node_mean->node = mean_running;
		node_var->node = var_running;
	}
	if (!mean_running) {
		FGraphNode *mean = tx;
		while (mean->operation.dimensions > 1)
			mean = fdiv_ci(freduce_sum(mean, 0),
						   (unsigned int)mean->operation.shape[0]);
		FGraphNode *var = fsub_g(tx, mean);
		while (var->operation.dimensions > 1)
			var = fdiv_ci(freduce_sum(var, 0),
						  (unsigned int)var->operation.shape[0]);
		incoming.push_back(new Variable(mean));
		incoming.push_back(new Variable(var));
		mean_running = mean;
		var_running = var;
	}
	FGraphNode *y = ftranspose(
		fadd_g(
			fmul_g(gamma, fdiv_g(fsub_g(tx, mean_running),
								 fsqrt_g(fadd_cf(
									 var_running,
									 std::numeric_limits<float>::epsilon())))),
			beta),
		transpositions1);
	output[0] = y;
};
