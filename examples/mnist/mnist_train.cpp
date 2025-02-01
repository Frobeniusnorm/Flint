#include "src/dl/layers.hpp"
#include "src/dl/model.hpp"
#include <cstring>
#include <flint/flint.h>
#include <flint/flint.hpp>
static Convolve *conv2D(unsigned long filters, unsigned long kernel_size,
						unsigned long units_in, unsigned long stride = 1,
						unsigned long padding = 0) {
	return new Convolve({1, 1}, {0, 0},
						Variable::fromGlorotUniform(
							{filters, kernel_size, kernel_size, stride}),
						Variable::fromConstant({filters}, 0.0));
}
static Connected* connected(unsigned long in, unsigned long out) {
	return new Connected(Variable::fromGlorotUniform({in, out}), 
						 Variable::fromConstant({1, out}, 0.0f));
}
int main() {
	FlintContext _(FLINT_BACKEND_ONLY_GPU, F_VERBOSE);
	fEnableEagerExecution();
	GraphModel *gm = GraphModel::sequential({
		conv2D(32, 3, 1),
		new Relu(),
		new MaxPool({2,2,1}, {2,2,1}, {0,0,0}),
		conv2D(64, 3, 32),
		new MaxPool({2,2,1}, {2,2,1}, {0,0,0}),
		new Flatten(),
		// TODO Dropout
		connected(1600, 10)
		// TODO SoftMax


	});
}
