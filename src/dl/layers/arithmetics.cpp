#include "../layers.hpp"
#include "flint.h"
// // //
// static data
// // //
int Add::add_no = 0;
int Connected::connected_no = 0;
// // //
// implementations
// // //
void Add::forward() {
#ifdef FLINT_DEBUG
	if (!((incoming.size() == 1 && incoming[0]->output.size() == 2) ||
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
static float diff(FGraphNode *a, FGraphNode *b) {

	// test if equal
	FGraphNode *diff = fabs_g(fsub(a, b));
	while (diff->operation.dimensions > 1)
		diff = fdiv_ci(freduce_sum(diff, 0), diff->operation.shape[0]);
	// to one element
	diff = fdiv_ci(freduce_sum(diff, 0), diff->operation.shape[0]);
	FResultData *res = fSyncMemory(fExecuteGraph(diff));
	float d = ((float *)res->data)[0];
	return d;
}
void Connected::forward() {
#ifdef FLINT_DEBUG
	if ((incoming.size() != 2 && incoming.size() != 3) ||
		incoming[0]->output.size() != 1 || incoming[1]->output.size() != 1 ||
		(incoming.size() == 3 && incoming[2]->output.size() != 1))
		flogging(
			F_ERROR,
			"Connected expects exactly two inputs, the input and the kernel");
#endif
	FGraphNode *img = incoming[0]->output[0];
	FGraphNode *kernel = incoming[1]->output[0];
	if (transposeA) {
		int transposition[img->operation.dimensions];
		for (int i = 0; i < img->operation.dimensions; i++)
			transposition[i] = i;
		transposition[img->operation.dimensions - 1] =
			img->operation.dimensions - 2;
		transposition[img->operation.dimensions - 2] =
			img->operation.dimensions - 1;
		img = ftranspose(img, transposition);
	}
	if (transposeB) {
		int transposition[img->operation.dimensions];
		for (int i = 0; i < img->operation.dimensions; i++)
			transposition[i] = i;
		transposition[img->operation.dimensions - 1] =
			img->operation.dimensions - 2;
		transposition[img->operation.dimensions - 2] =
			img->operation.dimensions - 1;
		kernel = ftranspose(kernel, transposition);
	}
	output[0] = fmatmul(img, kernel);
	if (incoming.size() == 3) {
		FGraphNode *bias = incoming[2]->output[0];
		output[0] = fadd(output[0], bias);
	}
}
