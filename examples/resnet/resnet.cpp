#include <flint/flint.h>
#include <flint/flint.hpp>
#include <flint/flint_helper.hpp>
#include <flint/model.hpp>
#include <iostream>
#include "imgnet_labels.h"
int main(int argc, char **argv) {
	FlintContext _(FLINT_BACKEND_ONLY_GPU);
  fEnableEagerExecution();
	if (argc != 3) {
		std::cout << "Usage: " << argv[0] << " <path to model> <path to image>"
				  << std::endl;
		return -1;
	}
	using namespace std;
	string model = argv[1];
	string image = argv[2];
	GraphModel& gm = *GraphModel::load_model(model);
	Tensor<float, 3> img =
		Flint::load_image(image).transpose({2, 1, 0}).transpose({0, 2, 1});
	Tensor<float, 4> batch = img.expand(0, 1);
	FGraphNode *out = gm(batch.get_graph_node());
	Tensor<float, 2> c(out);
	{
		float max_val = 0.0;
		int max_idx = 0;
		for (int i = 0; i < c.get_shape()[1]; i++) {
			if (c[0][i] > max_val) {
				max_val = c[0][i];
				max_idx = i;
			}
		}
		std::cout << imgnet_labels[max_idx] << std::endl;
	}
	// test
	std::string ser = gm.serialize_onnx();
	std::ofstream file;
	file.open("exported_model.onnx");
	file << ser;
	file.close();
}
