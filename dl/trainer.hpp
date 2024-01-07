
/* Copyright 2022 David Schwarzbeck

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */
#ifndef FLINT_TRAINER
#define FLINT_TRAINER
#include "losses.hpp"
#include "models.hpp"
#include "optimizers.hpp"
#include <cstdlib>
#include <flint/flint.h>
#include <flint/flint_helper.hpp>
#include <netinet/in.h>
#include <optional>
#include <semaphore>
#include <sys/socket.h>
#include <thread>
#include <unordered_map>

struct MetricInfo {
		int batch, epoch;
		size_t total_batches, total_epochs;
		double last_batch_error;
		double last_epoch_error;
		double last_validation_error;
};
struct MetricReporter {
		virtual ~MetricReporter() = default;
		virtual void report_batch(MetricInfo info) {
			std::cout << "\r\e[Kbatch error: " << std::setprecision(3)
					  << info.last_batch_error << " \e[1;96m";
			for (int k = 0; k < 15; k++) {
				if (k / 15.0 <= info.batch / (double)info.total_batches)
					std::cout << "―";
				else {
					std::cout << "\e[1;30m";
					for (int l = k; l < 15; l++)
						std::cout << "―";
					break;
				}
			}
			std::cout << "\033[0m" << std::flush;
		}
		virtual void report_epoch(MetricInfo info) {
			std::cout << "\n";
			flogging(F_INFO,
					 "Epoch #" + std::to_string(info.epoch) +
						 " error: " + std::to_string(info.last_epoch_error) +
						 " validation error: " +
						 std::to_string(info.last_validation_error));
		}
		virtual bool is_stop_signal() { return false; }
		virtual void report_finished() {}
		virtual void
		model_description(std::vector<std::string> layer_names,
						  std::vector<std::string> layer_descriptions,
						  std::vector<size_t> number_parameters,
						  std::string loss_fct, std::string optimizer_name,
						  std::string optimizer_desc) {
			this->layer_names = layer_names;
			this->layer_descriptions = layer_descriptions;
			this->number_parameters = number_parameters;
			this->loss_fct = loss_fct;
			this->optimizer_desc = optimizer_desc;
			this->optimizer_name = optimizer_name;
		}

	protected:
		std::vector<std::string> layer_names;
		std::vector<std::string> layer_descriptions;
		std::vector<size_t> number_parameters;
		std::string loss_fct, optimizer_name, optimizer_desc;
};
// TODO dataloader
template <typename T1, unsigned int n1, typename T2, unsigned int n2>
struct TrainingData {
		Tensor<T1, n1> X;
		Tensor<T2, n2> Y;
		std::optional<Tensor<T1, n1>> vX;
		std::optional<Tensor<T2, n2>> vY;
		TrainingData(Tensor<T1, n1> X, Tensor<T2, n2> Y) : X(X), Y(Y) {}
		TrainingData(Tensor<T1, n1> X, Tensor<T2, n2> Y, Tensor<T1, n1> vX,
					 Tensor<T2, n2> vY)
			: X(X), Y(Y), vX(vX), vY(vY) {}
};
template <typename T1, unsigned int n1, typename T2, unsigned int n2,
		  GenericLoss L, GenericLayer... T>
class Trainer {
		size_t epochs = 0;
		std::optional<double> to_error;
		MetricReporter *reporter = nullptr;
		MetricReporter default_reporter;
		MetricReporter &get_metric() {
			if (reporter)
				return *reporter;
			return default_reporter;
		}

	public:
		SequentialModel<T...> &model;
		TrainingData<T1, n1, T2, n2> &data;
		L loss;
		Trainer() = default;
		/**
		 * Trains the model with input data and the desired output.
		 * - `data` contains the input (`X`) and desired data (`Y`) and
		 *   optionally validation data, if it does after each epoch a
		 *   validation error is calculated.
		 * - `loss` The loss function to calculate the error between the actual
		 *   output and the desired one from the training data. Can be an
		 *   arbitrary class that implements the `GenericLoss` concept, some
		 *   implementations can be found in "losses.hpp".
		 * - `epochs` . */
		Trainer(SequentialModel<T...> &model,
				TrainingData<T1, n1, T2, n2> &data, L loss)
			: model(model), data(data), loss(loss) {
			using namespace std;
			const auto names = model.layer_names();
			const auto descriptions = model.layer_descriptions();
			const auto number_parameters = model.num_layer_parameters();
			this->default_reporter.model_description(
				vector<string>(names.begin(), names.end()),
				vector<string>(descriptions.begin(), descriptions.end()),
				vector<size_t>(number_parameters.begin(),
							   number_parameters.end()),
				loss.name(), model.optimizer(), model.optimizer_description());
		}

		/**
		 * Sets the maximum number of epochs after which the training should be
		 * stopped. The complete dataset is passed through the model per epoch
		 * (It is split into `batch_size` - configured in the `train` method -
		 * slices in the first dimension of the input data and each batch has to
		 * be passed through the model once per epoch)
		 */
		void max_epochs(int epochs) { this->epochs = epochs; }

		/**
		 * Sets the minimum epoch error after which the training should be
		 * stopped
		 */
		void stopping_error(double error) { this->to_error = error; }

		/**
		 * Sets the metric reporter (to print or display informations about the
		 * training process)
		 */
		void set_metric_reporter(MetricReporter *reporter) {
			this->reporter = reporter;
			using namespace std;
			const auto names = model.layer_names();
			const auto descriptions = model.layer_descriptions();
			const auto number_parameters = model.num_layer_parameters();
			reporter->model_description(
				vector<string>(names.begin(), names.end()),
				vector<string>(descriptions.begin(), descriptions.end()),
				vector<size_t>(number_parameters.begin(),
							   number_parameters.end()),
				loss.name(), model.optimizer(), model.optimizer_description());
		}
		/**
		 * Trains the model for the given batch size. A batch is a slice of the
		 * first imension of the input data. The input is shuffeled every epoch,
		 * which is important if your batch size is smaller then your input
		 * size. The weights of the model are optimized per batch that was
		 * passed through the model. Meaning small batch sizes lead to faster
		 * convergence (since more optimizations are executed) and lower memory
		 * consumption, but to more noise and variance, since each batch is only
		 * an approximation of the complete dataset. If training times and
		 * memory consumption don't matter we suggest full gradient descent
		 * (meaning `batch_size = input_size`), else finetune this value to your
		 * usecase.
		 */
		void train(int batch_size = 32) {
			const size_t batches = data.X.get_shape()[0];
			size_t number_batches = (size_t)ceil(batches / (double)batch_size);
			MetricInfo info_obj = {.batch = 0,
								   .epoch = 0,
								   .total_batches = number_batches,
								   .total_epochs = epochs,
								   .last_batch_error = 0,
								   .last_epoch_error = 0,
								   .last_validation_error = 0};
			if (data.Y.get_shape()[0] != batches)
				flogging(
					F_ERROR,
					"Input and Target Datas batch size does not correspond!");
			Tensor<long, 1> indices = Flint::arange(0, data.X.get_shape()[0]);
			for (int i = 0; i < epochs; i++) {
				// shuffle each epoch
				Tensor<T1, n1> sx = data.X.index(indices);
				Tensor<T2, n2> sy = data.Y.index(indices);
				indices = indices.permutate(0);
				indices.execute();
				double total_error = 0;
				for (size_t b = 0;
					 b < number_batches && !reporter->is_stop_signal(); b++) {
					size_t slice_to = (b + 1) * batch_size;
					if (slice_to > batches)
						slice_to = batches;
					if (b * batch_size == slice_to)
						break;
					// run batch and calculate error
					auto input =
						sx.slice(TensorRange(b * batch_size, slice_to));
					auto expected =
						sy.slice(TensorRange(b * batch_size, slice_to));
					input.execute();
					expected.execute();
					fStartGradientContext();
					auto output = model.forward_batch(input);
					auto error = loss.calculate_error(output, expected);
					fStopGradientContext();
					model.backward(error);
					double local_error = (double)(error.reduce_sum()[0]);
					total_error += local_error / number_batches;
					info_obj.last_batch_error = local_error;
					info_obj.batch = b + 1;
					get_metric().report_batch(info_obj);
				}
				info_obj.epoch = i + 1;
				info_obj.last_epoch_error = total_error;
				if (data.vX.has_value()) {
					auto output = model.forward_batch(data.vX.value());
					double val_error =
						(double)(loss.calculate_error(data.vY.value(), output)
									 .reduce_sum()[0]);
					info_obj.last_validation_error = val_error;
				}
				get_metric().report_epoch(info_obj);
				if (reporter->is_stop_signal())
					break;
			}
			get_metric().report_finished();
		}
};
/**
 * Sends the trainings data over a REST API for HTTP connections on the port
 * 5111. For a documentation of the API see `dl/visualization/README.md`
 */
class NetworkMetricReporter : public MetricReporter {
		std::thread thread;
		bool terminate;
		int socket_id;
		sockaddr_in sockaddr;
		std::vector<MetricInfo> batches, epochs;
		std::unordered_map<long, std::pair<long, long>> last_read;
		// controller states
		bool pause = false, stop = false, sent_all = false;
		std::binary_semaphore pause_lock;
		void open_connection() {
			socket_id = socket(AF_INET, SOCK_STREAM, 0);
			if (socket_id == -1)
				flogging(F_ERROR, "Could not open Web Socket!");
			sockaddr.sin_family = AF_INET;
			sockaddr.sin_addr.s_addr = INADDR_ANY;
			sockaddr.sin_port = htons(5111);
			if (bind(socket_id, (struct sockaddr *)&sockaddr,
					 sizeof(sockaddr)) < 0)
				flogging(F_ERROR, "Could not bind Web Socket! errno: " +
									  std::to_string(errno));
			if (listen(socket_id, 10) < 0) {
				flogging(F_ERROR, "Could not listen on Web Socket!");
			}
			flogging(F_VERBOSE, "Opened HTTP Metric Reporter on 5111");
		}

	public:
		void thread_routine() {
			using namespace std;
			while (!terminate || !stop) {
				size_t addrlen = sizeof(sockaddr);
				int connection = accept(socket_id, (struct sockaddr *)&sockaddr,
										(socklen_t *)&addrlen);
				if (connection < 0 && !terminate) {
					flogging(F_WARNING, "Connection Error!");
					continue;
				}
				char buffer[512];
				int no_bytes = read(connection, buffer, 512);
				if (no_bytes <= 0) {
					if (!terminate)
						flogging(F_WARNING,
								 "reading error: " + to_string(no_bytes) +
									 ", " + to_string(errno));
					close(connection);
					continue;
				}
				const string response = string(&buffer[0]);
				if (!response.starts_with("GET")) {
					flogging(F_WARNING, "Illegal Response!");
					close(connection);
					continue;
				}
				const string path =
					response.substr(4, response.find_first_of("\r\n ", 4) - 4);
				string packet;
				if (path == "/pause") {
					pause = true;
				} else if (path == "/play") {
					pause = false;
					pause_lock.release();
				} else if (path == "/stop") {
					pause = false;
					stop = true;
					pause_lock.release();
				} else if (path == "/describe") {
					packet = "{\"layers\":[";
					for (int i = 0; i < layer_names.size(); i++) {
						if (i != 0)
							packet += ",";
						packet += "{\"name\":\"" + layer_names[i] +
								  "\",\"description\":\"" +
								  layer_descriptions[i] + "\",\"no_params\":" +
								  to_string(number_parameters[i]) + "}";
					}
					packet += "],\"loss_fct\":\"" + loss_fct +
							  "\",\"optimizer\":{\"name\":\"" + optimizer_name +
							  "\", "
							  "\"description\":\"" +
							  optimizer_desc + "\"}}";

				} else {
					const long id = strtol(path.data() + 1, nullptr, 10);
					packet = "{";
					long last_read_batch = 0, last_read_epoch = 0;
					if (last_read.contains(id)) {
						const auto read_ids = last_read[id];
						last_read_batch = read_ids.first;
						last_read_epoch = read_ids.second;
						packet += "\"state\":";
						if (pause)
							packet += "\"pause\"";
						else
							packet += "\"play\"";
						packet += ",";
					}
					packet += "\"batches\":[";
					long total_batches = 0, total_epochs = 0;
					for (; last_read_batch < batches.size();
						 last_read_batch++) {
						MetricInfo batch = batches[last_read_batch];
						total_batches = batch.total_batches;
						packet +=
							"{\"batch\": " + to_string(batch.batch) +
							",\"error\": " + to_string(batch.last_batch_error) +
							"}";
						if (last_read_batch != batches.size() - 1)
							packet += ",";
					}
					packet += "], \"epochs\": [";
					for (; last_read_epoch < epochs.size(); last_read_epoch++) {
						MetricInfo epoch = epochs[last_read_epoch];
						total_epochs = epoch.total_epochs;
						packet +=
							"{\"epoch\": " + to_string(epoch.epoch) +
							",\"error\": " + to_string(epoch.last_epoch_error) +
							",\"validation_error\": " +
							to_string(epoch.last_validation_error) + "}";
						if (last_read_epoch != epochs.size() - 1)
							packet += ",";
					}
					last_read[id].first = last_read_batch;
					last_read[id].second = last_read_epoch;
					packet +=
						"], \"total_batches\": " + to_string(total_batches) +
						"}";
				}
				const string msg = "HTTP/1.1 200 OK\r\nServer: "
								   "Apache\r\nAccess-Control-Allow-Origin: "
								   "*\r\nAccept-Language: "
								   "en\r\nContent-Length:" +
								   to_string(packet.size()) +
								   "\r\nContent-Type: text/json\r\n\r\n";
				write(connection, msg.data(), msg.size());
				write(connection, packet.data(), packet.size());
				close(connection);
			}
		}
		NetworkMetricReporter() : pause_lock(0) {
			open_connection();
			terminate = false;
			thread = std::thread(&NetworkMetricReporter::thread_routine, this);
		}
		~NetworkMetricReporter() override {
			if (!terminate)
				report_finished();
		}
		NetworkMetricReporter(NetworkMetricReporter &&) = delete;
		NetworkMetricReporter(const NetworkMetricReporter &) = delete;
		NetworkMetricReporter &operator=(NetworkMetricReporter &&) = delete;
		NetworkMetricReporter &
		operator=(const NetworkMetricReporter &) = delete;
		void report_batch(MetricInfo info) override {
			if (pause) {
				pause_lock.acquire();
			}
			batches.push_back(info);
		}
		void report_finished() override {
			terminate = true;
			stop = true;
			shutdown(socket_id, SHUT_RD);
			close(socket_id);
			thread.join();
			int t = 1;
			setsockopt(socket_id, SOL_SOCKET, SO_REUSEADDR, &t, sizeof(int));
			flogging(F_VERBOSE, "Shutting down network");
		}
		void report_epoch(MetricInfo info) override { epochs.push_back(info); }
		bool is_stop_signal() override { return stop; }
};
#endif
