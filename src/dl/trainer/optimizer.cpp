#include "../trainer.hpp"
#include "flint.h"
#include <cmath>
#include <vector>
FGraphNode *Adam::optimize(FGraphNode *weight, FGraphNode *gradient) {
	if (m == nullptr) {
		if (weight->operation.data_type == F_FLOAT32) {
			m = fconstant_f(0.0, weight->operation.shape,
							weight->operation.dimensions);
			v = fconstant_f(0.0, weight->operation.shape,
							weight->operation.dimensions);
		} else {
			m = fconstant_d(0.0, weight->operation.shape,
							weight->operation.dimensions);
			v = fconstant_d(0.0, weight->operation.shape,
							weight->operation.dimensions);
		}
		m->reference_counter++;
		v->reference_counter++;
	}
	m = fadd_g(fmul_cf(m, b1), fmul_cf(gradient, (1 - b1)));
	v = fadd_g(fmul_cf(v, b2), fmul_g(gradient, fmul_cf(gradient, (1 - b2))));
	FGraphNode *mh = fdiv_cf(m, (1 - std::pow(b1, t)));
	FGraphNode *vh = fdiv_cf(v, (1 - std::pow(b2, t)));
	t += 1;
	return fsub_g(weight, fdiv_g(fmul_cf(mh, learning_rate),
								 fadd_cf(fsqrt_g(vh), epsilon)));
}
TrainingMetrics Trainer::train_epoch(size_t batch_size) {
	TrainingMetrics metrics;
	std::vector<FGraphNode *> weights(model->weights.size());
	for (int i = 0; i < weights.size(); i++)
		weights[i] = model->weights[i]->node;
	while (data->remaining_for_epoch()) {
		auto [in_nodes, out_nodes] = data->next_batch();
		fStartGradientContext();
		auto output = model->operator()(in_nodes);
		FGraphNode *error = nullptr;
		std::vector<FGraphNode *> errors(output.size());
		for (int i = 0; i < output.size(); i++) {
			errors[i] = loss->calculate_loss(output[i], out_nodes[i]);
			errors[i]->reference_counter++;
		}
		fStopGradientContext();
		std::vector<FGraphNode *> gradients(weights.size());
		fCalculateGradients(errors[0], weights.data(), weights.size(),
							gradients.data());
		for (int i = 1; i < output.size(); i++) {
			std::vector<FGraphNode *> local_gradients(weights.size());
			fCalculateGradients(errors[i], weights.data(), weights.size(),
								local_gradients.data());
			// add to gradients
			for (int j = 0; j < local_gradients.size(); j++)
				gradients[j] = fadd_g(gradients[j], local_gradients[j]);
		}
		if (output.size() > 1)
			for (int j = 0; j < gradients.size(); j++)
				gradients[j] =
					fdiv_ci(gradients[j], errors.size()); // averaging
		// create average loss for reporting
		double batch_loss = 0.0;
		for (int i = 0; i < output.size(); i++) {
			errors[i]->reference_counter--; // no longer needed
			while (errors[i]->operation.dimensions > 1) {
				errors[i] =
					freduce_sum(errors[i], errors[i]->operation.dimensions - 1);
			}
			errors[i] = fconvert(freduce_sum(errors[i], 0), F_FLOAT32);
			batch_loss += ((float*)fCalculateResult(errors[i])->result_data->data)[0];
		}
		// optimizing
		for (int j = 0; j < gradients.size(); j++) {
		optimizer->optimize(weights[j], gradients[j]);
		}
	}
	return metrics;
}
void Trainer::train(size_t epochs, size_t batch_size) {}
