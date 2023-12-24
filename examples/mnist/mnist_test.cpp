// #define FLINT_DL_PROFILE
#include "../../dl/flint_dl.hpp"
#include <flint/flint.h>
#include <flint/flint.hpp>
#include <flint/flint_helper.hpp>

int main(int argc, char **argv) {
	if (argc < 3) {
		std::cout << "Usage: " << argv[0] << " <path to model> <path to image>"
				  << std::endl;
	}
	FlintContext _(FLINT_BACKEND_ONLY_CPU, F_INFO);
	auto m = SequentialModel{
		Conv2D(1, 32, 8, std::array<unsigned int, 2>{3, 3}, SAME_PADDING),
		Relu(),
		Pooling<4>::max_pooling({3, 3, 1}, {2, 2, 1}, SAME_PADDING),
		Dropout(0.1),
		Flatten(),
		Connected(800, 80),
		Relu(),
		Dropout(0.1),
		Connected(80, 10),
		SoftMax()};
	m.load(std::string(argv[1]));
	Tensor<float, 3> img =
		Flint::load_image(std::string(argv[2]))
			.slice(TensorRange::MAX_SCOPE, TensorRange::MAX_SCOPE,
				   TensorRange(0, 1));
	std::cout << m.forward(img)() << std::endl;
}
