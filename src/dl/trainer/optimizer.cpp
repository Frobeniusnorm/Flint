#include "../trainer.hpp"
#include "flint.h"
#include <cmath>
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
	while (data->remaining_for_epoch()) {
		
	}
	return metrics;
}
void Trainer::train(size_t epochs, size_t batch_size) {
}
