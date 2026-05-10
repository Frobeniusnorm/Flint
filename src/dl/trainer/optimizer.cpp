#include "../trainer.hpp"
#include "flint.h"
#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>

static bool same_shape(const FGraphNode *a, const FGraphNode *b) {
	if (!a || !b)
		return false;
	if (a->operation.dimensions != b->operation.dimensions)
		return false;
	for (unsigned int i = 0; i < a->operation.dimensions; i++)
		if (a->operation.shape[i] != b->operation.shape[i])
			return false;
	return true;
}

static FGraphNode *materialize_graph(FGraphNode *node) {
	FGraphNode *evaluated = fCalculateResult(node);
	if (!evaluated || !evaluated->result_data) {
		flogging(F_ERROR, "Could not evaluate graph node.");
		return nullptr;
	}
	FGraphNode *res = fCreateGraph(
		evaluated->result_data->data, (int)evaluated->result_data->num_entries,
		evaluated->operation.data_type, evaluated->operation.shape,
		evaluated->operation.dimensions);
	if (!res)
		flogging(F_ERROR, "Could not materialize graph node.");
	return res;
}

static inline void free_graph_roots(std::vector<FGraphNode *> &nodes) {
	std::unordered_set<FGraphNode *> seen;
	for (FGraphNode *node : nodes) {
		if (node && seen.insert(node).second)
			fFreeGraph(node);
	}
	nodes.clear();
}

FGraphNode *Adam::optimize(FGraphNode *weight, FGraphNode *gradient) {
	if (!m || !same_shape(m, weight) ||
		m->operation.data_type != weight->operation.data_type) {
		if (m) {
			m->reference_counter--;
			fFreeGraph(m);
			m = nullptr;
		}
		if (v) {
			v->reference_counter--;
			fFreeGraph(v);
			v = nullptr;
		}
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
		t = 1;
	}

	FGraphNode *new_m_expr =
		fadd_g(fmul_cf(m, b1), fmul_cf(gradient, (1 - b1)));
	FGraphNode *new_v_expr =
		fadd_g(fmul_cf(v, b2), fmul_g(gradient, fmul_cf(gradient, (1 - b2))));
	FGraphNode *new_m = materialize_graph(new_m_expr);
	FGraphNode *new_v = materialize_graph(new_v_expr);
	fFreeGraph(new_m_expr);
	fFreeGraph(new_v_expr);
	new_m->reference_counter++;
	new_v->reference_counter++;
	m->reference_counter--;
	v->reference_counter--;
	fFreeGraph(m);
	fFreeGraph(v);
	m = new_m;
	v = new_v;

	FGraphNode *mh = fdiv_cf(m, (1 - std::pow(b1, t)));
	FGraphNode *vh = fdiv_cf(v, (1 - std::pow(b2, t)));
	t += 1;
	FGraphNode *new_weight_expr =
		fsub_g(weight, fdiv_g(fmul_cf(mh, learning_rate),
							  fadd_cf(fsqrt_g(vh), epsilon)));
	FGraphNode *new_weight = materialize_graph(new_weight_expr);
	fFreeGraph(new_weight_expr);
	return new_weight;
}
static void report_batch(int batch, int n_batch, float error = NAN,
						 bool first_print = true) {
	std::string output = "";
	if (!first_print) {
		output += "\r";
	}
	int num_digits_batch = ceil(log10(n_batch));
	output += "%0" + std::to_string(num_digits_batch) + "d/%d: [";
	int progress = (int)((batch / (double)n_batch) * 15);
	for (int i = 0; i < progress; i++) {
		output += "#";
	}
	for (int i = progress; i < 15; i++) {
		output += " ";
	}
	output += "], batch error: %f";
	printf(output.c_str(), batch, n_batch, error);
	fflush(stdout);
}
TrainingMetrics Trainer::train_epoch() {
	TrainingMetrics metrics;
	std::vector<FGraphNode *> weights(model->weights.size());
	for (int i = 0; i < weights.size(); i++)
		weights[i] = model->weights[i]->node;
	int total_batches = 0;
	metrics.training_loss = 0.0;
	report_batch(0, data->total_batches());
	do {
		for (FGraphNode *weight : weights)
			fMarkGradientVariable(weight);
		auto [in_nodes, out_nodes] = data->next_batch();
		fStartGradientContext();
		auto output = model->operator()(in_nodes);
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
			batch_loss +=
				((float *)fCalculateResult(errors[i])->result_data->data)[0];
		}
		// optimizing
		for (int j = 0; j < gradients.size(); j++) {
			FGraphNode *new_weight =
				optimizer->optimize(weights[j], gradients[j]);
			new_weight->reference_counter++;
			model->weights[j]->node->reference_counter--;
			fFreeGraph(model->weights[j]->node);
			model->weights[j]->node = new_weight;
			weights[j] = new_weight;
		}
		free_graph_roots(errors);
		metrics.training_loss += batch_loss;
		total_batches++;
		report_batch(total_batches, data->total_batches(), batch_loss, false);
	} while (data->remaining_for_epoch());
	metrics.training_loss /= total_batches;
	return metrics;
}
void Trainer::train(size_t epochs) {
	for (size_t i = 0; i < epochs; i++) {
		std::cout << "Training epoch " << (i + 1) << "/" << epochs << " ("
				  << (int)((i / (double)epochs) * 100.0) << "%)" << std::endl;
		TrainingMetrics metrics = train_epoch();
		// run validation
		auto [in_nodes, out_nodes] = data->validation_batch();
		auto output = model->operator()(in_nodes);
		double validation_error = 0.0;
		for (int i = 0; i < output.size(); i++) {
			FGraphNode *error = loss->calculate_loss(output[i], out_nodes[i]);
			while (error->operation.dimensions > 1) {
				error = freduce_sum(error, error->operation.dimensions - 1);
			}
			error = fconvert(freduce_sum(error, 0), F_FLOAT32);
			validation_error +=
				((float *)fCalculateResult(error)->result_data->data)[0];
			fFreeGraph(error);
		}
		metrics.validation_loss = validation_error;
		std::cout << "\ntraining loss: " << metrics.training_loss
				  << ", validation loss: " << validation_error << std::endl;
	}
}
FGraphNode *CrossEntropyLoss::calculate_loss(FGraphNode *out, FGraphNode *exp) {
	const int n = out->operation.dimensions;
	auto pred = fmin_cd(fmax_cd(out, 1e-7), 1 - 1e-7);
	auto t1 = (fmul(exp, fneg(flog(pred))));
	while (t1->operation.dimensions > 1)
		t1 = freduce_sum(t1, 1);
	size_t total_size = 1;
	for (unsigned int i = 0; i < n - 1; i++)
		total_size *= out->operation.shape[i];
	return fdiv_cd(t1, (double)total_size);
}
