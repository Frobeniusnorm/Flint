#include "../../src/dl/model.hpp"
#include "../../src/dl/trainer.hpp"
#include "../../flint.h"
#include "../../flint.hpp"
int main() {
	FlintContext _(FLINT_BACKEND_ONLY_GPU, F_WARNING);
	GraphModel *gm = GraphModel::builder()
						 .conv2d(32, {3, 3}, 1, FlintDL::ActivationKind::Relu)
						 .maxpool2d({2, 2}, std::array<unsigned int, 2>{2, 2})
						 .conv2d(64, {3, 3}, 32, FlintDL::ActivationKind::Relu)
						 .maxpool2d({2, 2}, std::array<unsigned int, 2>{2, 2})
						 .flatten()
						 .dropout(0.5f)
						 .dense(1600, 10, FlintDL::ActivationKind::Softmax)
						 .build();
	IDXFormatLoader idx(512, "train-images-idx3-ubyte",
						"train-labels-idx1-ubyte", "t10k-images-idx3-ubyte",
						"t10k-labels-idx1-ubyte");
	Adam adam;
	CrossEntropyLoss loss;
	Trainer t(gm);
	t.set_data_loader(&idx);
	t.set_optimizer(&adam);
	t.set_loss(&loss);
	t.train(10);
}
