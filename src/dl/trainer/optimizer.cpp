#include "../trainer.hpp"
#include "flint.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
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

static std::string json_escape(const std::string &in) {
	std::string escaped;
	escaped.reserve(in.size());
	for (char c : in) {
		switch (c) {
		case '\"':
			escaped += "\\\"";
			break;
		case '\\':
			escaped += "\\\\";
			break;
		case '\n':
			escaped += "\\n";
			break;
		case '\r':
			escaped += "\\r";
			break;
		case '\t':
			escaped += "\\t";
			break;
		default:
			escaped += c;
			break;
		}
	}
	return escaped;
}

static std::string layer_description(const LayerGraph *layer) {
	if (dynamic_cast<const Relu *>(layer))
		return "Relu Activation Layer";
	if (dynamic_cast<const Softmax *>(layer))
		return "Softmax Activation Layer";
	if (dynamic_cast<const Dropout *>(layer))
		return "Dropout Layer";
	if (dynamic_cast<const Flatten *>(layer))
		return "Flatten Layer";
	if (dynamic_cast<const Add *>(layer))
		return "Add Layer";
	if (dynamic_cast<const Convolve *>(layer))
		return "Convolution Layer";
	if (dynamic_cast<const MaxPool *>(layer))
		return "MaxPooling Layer";
	if (dynamic_cast<const AvgPool *>(layer))
		return "AveragePooling Layer";
	if (dynamic_cast<const GlobalAvgPool *>(layer))
		return "GlobalAveragePooling Layer";
	if (dynamic_cast<const BatchNorm *>(layer))
		return "BatchNormalization Layer";
	if (dynamic_cast<const Connected *>(layer))
		return "Dense Layer";
	return "Layer";
}

static size_t graph_entries(const FGraphNode *node) {
	if (!node)
		return 0;
	size_t entries = 1;
	for (unsigned int i = 0; i < node->operation.dimensions; i++)
		entries *= node->operation.shape[i];
	return entries;
}

static size_t layer_parameters(const LayerGraph *layer) {
	size_t total = 0;
	for (LayerGraph *in : layer->incoming) {
		const Variable *var = dynamic_cast<const Variable *>(in);
		if (var)
			total += graph_entries(var->node);
	}
	return total;
}

static void collect_layers_postorder(
	LayerGraph *layer, std::unordered_set<LayerGraph *> &visited,
	std::vector<LayerGraph *> &ordered_layers) {
	if (!layer || !visited.insert(layer).second)
		return;
	for (LayerGraph *in : layer->incoming)
		collect_layers_postorder(in, visited, ordered_layers);
	if (!dynamic_cast<InputNode *>(layer) && !dynamic_cast<Variable *>(layer) &&
		!dynamic_cast<ConstantNode *>(layer))
		ordered_layers.push_back(layer);
}

bool ReporterControlInformation::stop_signal() const {
	return stop_signal_state.load();
}

void ReporterControlInformation::set_stop_signal(bool stop) {
	stop_signal_state.store(stop);
}

bool ReporterControlInformation::profiling() const {
	return profiling_state.load();
}

void ReporterControlInformation::set_profiling(bool profiling) {
	profiling_state.store(profiling);
}

MetricReporter::MetricReporter()
	: control_info(std::make_shared<ReporterControlInformation>()) {}

MetricReporter::MetricReporter(
	std::shared_ptr<ControlInformation> control_information)
	: control_info(control_information
					   ? std::move(control_information)
					   : std::make_shared<ReporterControlInformation>()) {}

ControlInformation &MetricReporter::control_information() const {
	return *control_info;
}

void MetricReporter::model_description(
	std::vector<std::string> layer_names,
	std::vector<std::string> layer_descriptions,
	std::vector<size_t> number_parameters, std::string loss_fct,
	std::string optimizer_name, std::string optimizer_desc) {
	this->layer_names = std::move(layer_names);
	this->layer_descriptions = std::move(layer_descriptions);
	this->number_parameters = std::move(number_parameters);
	this->loss_fct = std::move(loss_fct);
	this->optimizer_name = std::move(optimizer_name);
	this->optimizer_desc = std::move(optimizer_desc);
}

void CLIReporter::report_batch(const MetricInfo &info) {
	const int total_batches = std::max(1, (int)info.total_batches);
	const int width = std::max(1, (int)std::to_string(total_batches).size());
	const int progress = std::clamp(
		(int)((info.batch / (double)total_batches) * 15.0), 0, 15);
	if (!first_print)
		std::cout << "\r";
	std::cout << std::setfill('0') << std::setw(width) << info.batch << "/"
			  << total_batches << ": [";
	for (int i = 0; i < progress; i++)
		std::cout << "#";
	for (int i = progress; i < 15; i++)
		std::cout << " ";
	std::cout << "], batch error: " << std::setprecision(6)
			  << info.last_batch_error << std::flush;
	first_print = false;
}

void CLIReporter::report_epoch(const MetricInfo &info) {
	if (!first_print)
		std::cout << "\n";
	flogging(F_INFO, "Epoch #" + std::to_string(info.epoch) +
						 " error: " + std::to_string(info.last_epoch_error) +
						 " validation error: " +
						 std::to_string(info.last_validation_error));
	first_print = true;
}

void CLIReporter::report_finished() {
	if (!first_print)
		std::cout << std::endl;
	first_print = true;
}

void NetworkMetricReporter::open_connection() {
	socket_id = socket(AF_INET, SOCK_STREAM, 0);
	if (socket_id == -1)
		flogging(F_ERROR, "Could not open Web Socket!");
	int reuse = 1;
	setsockopt(socket_id, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
	std::memset(&socket_address, 0, sizeof(socket_address));
	socket_address.sin_family = AF_INET;
	socket_address.sin_addr.s_addr = INADDR_ANY;
	socket_address.sin_port = htons(port);
	if (bind(socket_id, (struct sockaddr *)&socket_address,
			 sizeof(socket_address)) < 0)
		flogging(F_ERROR,
				 "Could not bind Web Socket! errno: " +
					 std::to_string(errno));
	if (listen(socket_id, 10) < 0)
		flogging(F_ERROR, "Could not listen on Web Socket!");
	flogging(F_VERBOSE, "Opened HTTP Metric Reporter on " + std::to_string(port));
}

void NetworkMetricReporter::wait_if_paused() {
	std::unique_lock<std::mutex> lock(pause_lock);
	pause_cv.wait(lock, [&]() {
		return !pause || terminate.load() || control_information().stop_signal();
	});
}

void NetworkMetricReporter::thread_routine() {
	while (!terminate.load()) {
		socklen_t addrlen = sizeof(socket_address);
		int connection = accept(socket_id, (struct sockaddr *)&socket_address,
								&addrlen);
		if (connection < 0) {
			if (!terminate.load())
				flogging(F_WARNING, "Connection Error!");
			continue;
		}
		char buffer[1024] = {0};
		int no_bytes = read(connection, buffer, sizeof(buffer) - 1);
		if (no_bytes <= 0) {
			if (!terminate.load())
				flogging(F_WARNING,
						 "reading error: " + std::to_string(no_bytes) + ", " +
							 std::to_string(errno));
			close(connection);
			continue;
		}
		const std::string request(buffer, no_bytes);
		if (!request.starts_with("GET ")) {
			flogging(F_WARNING, "Illegal Response!");
			close(connection);
			continue;
		}
		const size_t path_end = request.find_first_of("\r\n ", 4);
		const std::string path = request.substr(4, path_end - 4);
		std::string packet = "{}";
		if (path == "/pause") {
			std::lock_guard<std::mutex> lock(pause_lock);
			pause = true;
		} else if (path == "/play" || path == "/start") {
			{
				std::lock_guard<std::mutex> lock(pause_lock);
				pause = false;
			}
			pause_cv.notify_all();
		} else if (path == "/stop") {
			{
				std::lock_guard<std::mutex> lock(pause_lock);
				pause = false;
			}
			control_information().set_stop_signal(true);
			pause_cv.notify_all();
		} else if (path == "/describe") {
			std::lock_guard<std::mutex> lock(data_lock);
			packet = "{\"layers\":[";
			for (size_t i = 0; i < layer_names.size(); i++) {
				if (i != 0)
					packet += ",";
				packet += "{\"name\":\"" + json_escape(layer_names[i]) +
						  "\",\"description\":\"" +
						  json_escape(layer_descriptions[i]) +
						  "\",\"no_params\":" +
						  std::to_string(number_parameters[i]) + "}";
			}
			packet += "],\"loss_fct\":\"" + json_escape(loss_fct) +
					  "\",\"optimizer\":{\"name\":\"" +
					  json_escape(optimizer_name) + "\",\"description\":\"" +
					  json_escape(optimizer_desc) + "\"}}";
		} else if (path == "/start_profiling") {
			control_information().set_profiling(true);
		} else if (path == "/stop_profiling") {
			control_information().set_profiling(false);
		} else {
			long id = 0;
			if (path.size() > 1)
				id = strtol(path.data() + 1, nullptr, 10);
			std::lock_guard<std::mutex> lock(data_lock);
			packet = "{";
			long last_read_batch = 0, last_read_epoch = 0;
			if (last_read.contains(id)) {
				const auto read_ids = last_read[id];
				last_read_batch = read_ids.first;
				last_read_epoch = read_ids.second;
				packet += "\"state\":";
				{
					std::lock_guard<std::mutex> pause_guard(pause_lock);
					packet += pause ? "\"pause\"" : "\"play\"";
				}
				packet += ",";
			}
			packet += "\"profiling\":";
			packet += control_information().profiling() ? "true" : "false";
			packet += ",\"batches\":[";
			long total_batches = 0;
			bool read_something = false;
			for (; last_read_batch < (long)batches.size(); last_read_batch++) {
				read_something = true;
				const MetricInfo &batch = batches[last_read_batch];
				total_batches = batch.total_batches;
				packet += "{\"batch\":" + std::to_string(batch.batch) +
						  ",\"error\":" +
						  std::to_string(batch.last_batch_error) + "}";
				if (last_read_batch != (long)batches.size() - 1)
					packet += ",";
			}
			packet += "],\"epochs\":[";
			for (; last_read_epoch < (long)epochs.size(); last_read_epoch++) {
				read_something = true;
				const MetricInfo &epoch = epochs[last_read_epoch];
				packet += "{\"epoch\":" + std::to_string(epoch.epoch) +
						  ",\"error\":" +
						  std::to_string(epoch.last_epoch_error) +
						  ",\"validation_error\":" +
						  std::to_string(epoch.last_validation_error) + "}";
				if (last_read_epoch != (long)epochs.size() - 1)
					packet += ",";
			}
			last_read[id] = {last_read_batch, last_read_epoch};
			packet += "],\"total_batches\":" + std::to_string(total_batches);
			if (read_something && !batches.empty() &&
				!batches.back().time_per_layer_ns.empty()) {
				packet += ",\"profiling_data\":{\"forward\":[";
				const MetricInfo &last_batch = batches.back();
				for (size_t i = 0; i < last_batch.time_per_layer_ns.size(); i++) {
					if (i != 0)
						packet += ",";
					packet += "{\"name\":\"" +
							  json_escape(last_batch.time_per_layer_ns[i].first) +
							  "\",\"time\":" +
							  std::to_string(last_batch.time_per_layer_ns[i].second) +
							  "}";
				}
				packet += "],\"gradient\":" +
						  std::to_string(last_batch.gradient_time_ns) + "}";
			}
			packet += "}";
		}
		const std::string msg = "HTTP/1.1 200 OK\r\nServer: Flint\r\n"
								"Access-Control-Allow-Origin: *\r\n"
								"Content-Length:" +
								std::to_string(packet.size()) +
								"\r\nContent-Type: text/json\r\n\r\n";
		write(connection, msg.data(), msg.size());
		write(connection, packet.data(), packet.size());
		close(connection);
	}
}

NetworkMetricReporter::NetworkMetricReporter(unsigned short port)
	: port(port) {
	open_connection();
	thread = std::thread(&NetworkMetricReporter::thread_routine, this);
}

NetworkMetricReporter::~NetworkMetricReporter() {
	report_finished();
}

void NetworkMetricReporter::report_batch(const MetricInfo &info) {
	wait_if_paused();
	std::lock_guard<std::mutex> lock(data_lock);
	batches.push_back(info);
}

void NetworkMetricReporter::report_epoch(const MetricInfo &info) {
	std::lock_guard<std::mutex> lock(data_lock);
	epochs.push_back(info);
}

void NetworkMetricReporter::report_finished() {
	std::lock_guard<std::mutex> finish_guard(finish_lock);
	if (has_finished)
		return;
	has_finished = true;
	terminate = true;
	control_information().set_stop_signal(true);
	{
		std::lock_guard<std::mutex> lock(pause_lock);
		pause = false;
	}
	pause_cv.notify_all();
	if (socket_id >= 0) {
		shutdown(socket_id, SHUT_RDWR);
		close(socket_id);
		socket_id = -1;
	}
	if (thread.joinable())
		thread.join();
	flogging(F_VERBOSE, "Shutting down network reporter.");
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

std::string Adam::description() const {
	std::ostringstream description;
	description << "learning rate: " << learning_rate << ", beta1: " << b1
				<< ", beta2: " << b2 << ", epsilon: " << epsilon;
	return description.str();
}

std::string CrossEntropyLoss::description() const {
	return "Categorical cross entropy: sum(-expected * log(actual)).";
}

MetricReporter &Trainer::get_metric_reporter() {
	if (reporter)
		return *reporter;
	return default_reporter;
}

void Trainer::refresh_metric_reporter(MetricReporter &metric) {
	std::vector<std::string> names;
	std::vector<std::string> descriptions;
	std::vector<size_t> parameters;
	if (model) {
		std::unordered_set<LayerGraph *> visited;
		std::vector<LayerGraph *> ordered_layers;
		for (LayerGraph *out : model->output)
			collect_layers_postorder(out, visited, ordered_layers);
		for (LayerGraph *layer : ordered_layers) {
			names.push_back(layer->name);
			descriptions.push_back(layer_description(layer));
			parameters.push_back(layer_parameters(layer));
		}
	}
	metric.model_description(
		names, descriptions, parameters,
		loss ? loss->name() : "Unknown loss function",
		optimizer ? optimizer->name() : "Unknown optimizer",
		optimizer ? optimizer->description() : "No optimizer configured.");
}

void Trainer::refresh_metric_reporters() {
	refresh_metric_reporter(default_reporter);
	if (reporter)
		refresh_metric_reporter(*reporter);
}

TrainingMetrics Trainer::train_epoch() {
	TrainingMetrics metrics = {.is_epoch = true,
							   .training_loss = 0.0,
							   .validation_loss = 0.0,
							   .training_time_ms = 0.0,
							   .validation_time_ms = 0.0,
							   .avg_batch_time_ms = 0.0};
	if (!model || !data || !optimizer || !loss) {
		flogging(F_ERROR, "Trainer is missing model, data, optimizer or loss.");
		return metrics;
	}
	std::vector<FGraphNode *> weights(model->weights.size());
	for (size_t i = 0; i < weights.size(); i++)
		weights[i] = model->weights[i]->node;
	size_t trained_batches = 0;
	std::map<LayerGraph *, long> aggregated_layer_time_ns;
	MetricReporter &metric_reporter = get_metric_reporter();
	if (metric_reporter.control_information().stop_signal())
		return metrics;
	do {
		if (metric_reporter.control_information().stop_signal())
			break;
		const bool collect_profiling =
			metric_reporter.control_information().profiling();
		const auto batch_start = std::chrono::steady_clock::now();
		for (FGraphNode *weight : weights)
			fMarkGradientVariable(weight);
		auto [in_nodes, out_nodes] = data->next_batch();
		fStartGradientContext();
		std::map<LayerGraph *, long> batch_layer_time_ns;
		auto output = collect_profiling
						  ? model->operator()(in_nodes, std::ref(batch_layer_time_ns))
						  : model->operator()(in_nodes);
		std::vector<FGraphNode *> errors(output.size());
		for (size_t i = 0; i < output.size(); i++) {
			errors[i] = loss->calculate_loss(output[i], out_nodes[i]);
			errors[i]->reference_counter++;
		}
		fStopGradientContext();
		const auto gradient_start = std::chrono::steady_clock::now();
		std::vector<FGraphNode *> gradients(weights.size());
		fCalculateGradients(errors[0], weights.data(), weights.size(),
							gradients.data());
		for (size_t i = 1; i < output.size(); i++) {
			std::vector<FGraphNode *> local_gradients(weights.size());
			fCalculateGradients(errors[i], weights.data(), weights.size(),
								local_gradients.data());
			for (size_t j = 0; j < local_gradients.size(); j++)
				gradients[j] = fadd_g(gradients[j], local_gradients[j]);
		}
		if (output.size() > 1)
			for (size_t j = 0; j < gradients.size(); j++)
				gradients[j] =
					fdiv_ci(gradients[j], errors.size()); // averaging
		const auto gradient_end = std::chrono::steady_clock::now();
		const double gradient_time_ns =
			std::chrono::duration_cast<std::chrono::nanoseconds>(gradient_end -
																 gradient_start)
				.count();
		double batch_loss = 0.0;
		for (size_t i = 0; i < output.size(); i++) {
			errors[i]->reference_counter--;
			while (errors[i]->operation.dimensions > 1) {
				errors[i] =
					freduce_sum(errors[i], errors[i]->operation.dimensions - 1);
			}
			errors[i] = fconvert(freduce_sum(errors[i], 0), F_FLOAT32);
			batch_loss +=
				((float *)fCalculateResult(errors[i])->result_data->data)[0];
		}
		for (size_t j = 0; j < gradients.size(); j++) {
			FGraphNode *new_weight =
				optimizer->optimize(weights[j], gradients[j]);
			new_weight->reference_counter++;
			model->weights[j]->node->reference_counter--;
			fFreeGraph(model->weights[j]->node);
			model->weights[j]->node = new_weight;
			weights[j] = new_weight;
		}
		free_graph_roots(errors);
		const auto batch_end = std::chrono::steady_clock::now();
		metrics.training_time_ms +=
			std::chrono::duration<double, std::milli>(batch_end - batch_start)
				.count();
		metrics.training_loss += batch_loss;
		trained_batches++;
		if (collect_profiling) {
			for (const auto &[layer, time_ns] : batch_layer_time_ns)
				aggregated_layer_time_ns[layer] += time_ns;
		}
		MetricInfo info = {.batch = (int)trained_batches,
						   .epoch = (int)active_epoch,
						   .total_batches = data->total_batches(),
						   .total_epochs = active_total_epochs,
						   .last_batch_error = batch_loss,
						   .gradient_time_ns = gradient_time_ns};
		if (collect_profiling) {
			info.time_per_layer_ns.reserve(batch_layer_time_ns.size());
			for (const auto &[layer, time_ns] : batch_layer_time_ns) {
				if (layer)
					info.time_per_layer_ns.push_back(
						{layer->name, static_cast<double>(time_ns)});
			}
		}
		metric_reporter.report_batch(info);
	} while (data->remaining_for_epoch());
	if (trained_batches > 0) {
		metrics.training_loss /= trained_batches;
		metrics.avg_batch_time_ms = metrics.training_time_ms / trained_batches;
		if (!aggregated_layer_time_ns.empty()) {
			metrics.avg_batch_time_per_layer_ms.reserve(
				aggregated_layer_time_ns.size());
			for (const auto &[layer, time_ns] : aggregated_layer_time_ns) {
				if (!layer)
					continue;
				metrics.avg_batch_time_per_layer_ms.push_back(
					{layer->name,
					 (time_ns / 1000000.0) / static_cast<double>(trained_batches)});
			}
		}
	}
	return metrics;
}

void Trainer::train(size_t epochs) {
	if (!model || !data || !optimizer || !loss)
		return flogging(F_ERROR,
					   "Trainer is missing model, data, optimizer or loss.");
	MetricReporter &metric_reporter = get_metric_reporter();
	metric_reporter.control_information().set_stop_signal(false);
	active_total_epochs = epochs;
	for (size_t i = 0; i < epochs; i++) {
		active_epoch = i + 1;
		TrainingMetrics metrics = train_epoch();
		const auto validation_start = std::chrono::steady_clock::now();
		auto [in_nodes, out_nodes] = data->validation_batch();
		auto output = model->operator()(in_nodes);
		double validation_error = 0.0;
		for (size_t j = 0; j < output.size(); j++) {
			FGraphNode *error = loss->calculate_loss(output[j], out_nodes[j]);
			while (error->operation.dimensions > 1)
				error = freduce_sum(error, error->operation.dimensions - 1);
			error = fconvert(freduce_sum(error, 0), F_FLOAT32);
			validation_error +=
				((float *)fCalculateResult(error)->result_data->data)[0];
			fFreeGraph(error);
		}
		metrics.validation_loss = validation_error;
		metrics.validation_time_ms =
			std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - validation_start)
				.count();
		MetricInfo epoch_info = {.epoch = (int)(i + 1),
								 .total_batches = data->total_batches(),
								 .total_epochs = epochs,
								 .last_epoch_error = metrics.training_loss,
								 .last_validation_error = validation_error};
		metric_reporter.report_epoch(epoch_info);
		if (metric_reporter.control_information().stop_signal())
			break;
		if (early_stopping_error.has_value() &&
			validation_error <= early_stopping_error.value())
			break;
	}
	metric_reporter.report_finished();
	active_epoch = 0;
	active_total_epochs = 0;
}

FGraphNode *CrossEntropyLoss::calculate_loss(FGraphNode *out, FGraphNode *exp) {
	const int n = out->operation.dimensions;
	auto pred = fmin_cd(fmax_cd(out, 1e-7), 1 - 1e-7);
	auto t1 = (fmul(exp, fneg(flog(pred))));
	while (t1->operation.dimensions > 1)
		t1 = freduce_sum(t1, 1);
	size_t total_size = 1;
	for (unsigned int i = 0; i < (unsigned int)(n - 1); i++)
		total_size *= out->operation.shape[i];
	return fdiv_cd(t1, (double)total_size);
}
