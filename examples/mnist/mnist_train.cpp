#include "../../flint.h"
#include "../../flint.hpp"
#include "../../src/dl/model.hpp"
#include "../../src/dl/trainer.hpp"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>

/** Reports wall time per batch so the example can be used to tune the
 * GPU backend. Stops training early if a batch limit was given. */
class BenchReporter : public MetricReporter {
		std::chrono::steady_clock::time_point last =
			std::chrono::steady_clock::now();
		double total_ms = 0;
		int measured = 0;
		int max_batches;

	public:
		explicit BenchReporter(int max_batches) : max_batches(max_batches) {}
		void report_batch(const MetricInfo &info) override {
			const auto now = std::chrono::steady_clock::now();
			const double ms =
				std::chrono::duration<double, std::milli>(now - last).count();
			last = now;
			// the first batch includes kernel compilation, don't average it in
			if (info.batch > 1) {
				total_ms += ms;
				measured++;
			}
			std::cout << "batch " << std::setw(4) << info.batch << "/"
					  << info.total_batches << "  " << std::fixed
					  << std::setprecision(1) << std::setw(8) << ms << " ms"
					  << "  (gradient " << std::setw(7)
					  << info.gradient_time_ns / 1e6 << " ms)"
					  << "  error " << std::setprecision(5)
					  << info.last_batch_error << std::endl;
			if (max_batches > 0 && info.batch >= max_batches)
				control_information().set_stop_signal(true);
		}
		void report_epoch(const MetricInfo &info) override {
			flogging(F_INFO, "epoch " + std::to_string(info.epoch) + " error " +
								 std::to_string(info.last_epoch_error) +
								 " validation " +
								 std::to_string(info.last_validation_error));
		}
		void report_finished() override {
			if (measured)
				flogging(F_INFO, "average " +
									 std::to_string(total_ms / measured) +
									 " ms per batch over " +
									 std::to_string(measured) + " batches");
		}
};

int main(int argc, char **argv) {
	int epochs = 10, max_batches = 0, batch_size = 512;
	FLogType log_level = F_WARNING;
	for (int i = 1; i < argc; i++) {
		if (std::strcmp(argv[i], "--debug") == 0)
			log_level = F_DEBUG;
		else if (i + 1 >= argc)
			continue;
		else if (std::strcmp(argv[i], "--epochs") == 0)
			epochs = std::atoi(argv[++i]);
		else if (std::strcmp(argv[i], "--batches") == 0)
			max_batches = std::atoi(argv[++i]);
		else if (std::strcmp(argv[i], "--batch-size") == 0)
			batch_size = std::atoi(argv[++i]);
	}
	FlintContext _(FLINT_BACKEND_ONLY_GPU, log_level);
	GraphModel *gm = GraphModel::builder()
						 .conv2d(32, {3, 3}, 1, FlintDL::ActivationKind::Relu)
						 .maxpool2d({2, 2}, std::array<unsigned int, 2>{2, 2})
						 .conv2d(64, {3, 3}, 32, FlintDL::ActivationKind::Relu)
						 .maxpool2d({2, 2}, std::array<unsigned int, 2>{2, 2})
						 .flatten()
						 .dropout(0.5f)
						 .dense(1600, 10, FlintDL::ActivationKind::Softmax)
						 .build();
	IDXFormatLoader idx(batch_size, "train-images-idx3-ubyte",
						"train-labels-idx1-ubyte", "t10k-images-idx3-ubyte",
						"t10k-labels-idx1-ubyte");
	Adam adam;
	CrossEntropyLoss loss;
	BenchReporter reporter(max_batches);
	Trainer t(gm);
	t.set_metric_reporter(&reporter);
	t.set_data_loader(&idx);
	t.set_optimizer(&adam);
	t.set_loss(&loss);
	t.train(epochs);
	if (max_batches == 0) {
		std::ofstream out_file("model.onnx", std::ios::out | std::ios::trunc);
		out_file << gm->serialize_onnx();
	}
}
