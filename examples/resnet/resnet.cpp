#include <flint/flint.h>
#include <flint/flint.hpp>
#include <flint/model.hpp>
#include <iostream>
int main(int argc, char **argv) {
	if (argc != 3) {
		std::cout << "Usage: " << argv[0] << " <path to model> <path to image>"
				  << std::endl;
		return -1;
	}
	using namespace std;
	string model = argv[1];
	string image = argv[2];
	GraphModel gm = GraphModel::load_model(model);
	Tensor<float, 3> img = Flint::load_image(image);
	Tensor<float, 4> batch = img.expand(0, 1);
	FGraphNode *out = gm(batch.get_graph_node());
	Tensor<float, 2> c(out);
	std::cout << c << std::endl;
}
