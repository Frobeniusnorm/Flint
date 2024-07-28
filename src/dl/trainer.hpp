#ifndef FLINT_DL_TRAINER
#define FLINT_DL_TRAINER

#include "../../flint.h"
#include "model.hpp"
#include <optional>
#include <vector>
/**
 * Loads the Data for the training process
 */
struct DataLoader {
		size_t batch_size;
		DataLoader(size_t batch_size) : batch_size(batch_size) {}
		virtual ~DataLoader() {}
		/**
		 * Loads the next batch and returns it as a pair of model input and
		 * expected output. I.e. the returned pair is a tuple, where the first
		 * entry describes the input values for the model (each entry in the
		 * vector is a batch-sized input for the model, the vector is used for
		 * models that have multiple inputs. If your model just has one, return
		 * a 1-element vector) and the second the output values that are
		 * expected.
		 */
		virtual std::pair<std::vector<FGraphNode *>, std::vector<FGraphNode *>>
		next_batch() = 0;
		/**
		 * Used to determine how many elements (not batches!) are still to be
		 * processed to finish the epoch. Return 0 if the epoch is finished.
		 * Used for metrics and determining if validation can be run and the
		 * next epoch started.
		 */
		virtual size_t remaining_for_epoch() = 0;
		/**
		 * Returns the data for the validation. Same semantic as for
		 * `next_batch`.
		 */
		virtual std::pair<std::vector<FGraphNode *>, std::vector<FGraphNode *>>
		validation_batch() = 0;
		/** Return the complete training dataset, used for testing after
		 * training */
		virtual std::pair<std::vector<FGraphNode *>, std::vector<FGraphNode *>>
		testing_data() = 0;
};
/**
 * Loads IDX formattet ubyte files like the ones used for the MNIST dataset.
 */
struct IDXFormatLoader : public DataLoader {
		/** Sets the batch size, the paths to the train and test data and the
		 * validation percentage. The validation percentage is the percentage of
		 * the training data that is split to validate the error after each
		 * training epoch. */
		IDXFormatLoader(size_t batch_size, std::string train_images_path,
						std::string train_labels_path,
						std::string test_images_path = "",
						std::string test_labels_path = "",
						double validation_percentage = 0.15)
			: DataLoader(batch_size), train_images_path(train_images_path),
			  train_labels_path(train_images_path),
			  test_images_path(test_images_path),
			  test_labels_path(test_labels_path),
			  validation_percentage(validation_percentage) {
			prefetch_data();
		}
		std::pair<std::vector<FGraphNode *>, std::vector<FGraphNode *>>
		next_batch() override;
		std::pair<std::vector<FGraphNode *>, std::vector<FGraphNode *>>
		validation_batch() override;
		size_t remaining_for_epoch() override;
		std::pair<std::vector<FGraphNode *>, std::vector<FGraphNode *>>
		testing_data() override;

	private:
		std::string train_images_path, train_labels_path;
		std::string test_images_path, test_labels_path;
		double validation_percentage;
		// TODO dont prefetch data but load lazy when needed if > 6GB or
		// something
		FGraphNode *training_data, *validation_data, *test_data;
		FGraphNode *training_labels, *validation_labels, *test_labels;
		void prefetch_data();
};
/**
 * Interface to optimize variables.
 * For each Variable an Optimizer is created and managed.
 */
struct Optimizer {
		/**
		 * Updates the weight regarding its gradient or derivation.
		 * `weight` is the variable and `gradient` the gradient.
		 * Returns the new variable, the old one will be replaced.
		 * Do not set the reference counter as this is done by the trainer.
		 */
		virtual FGraphNode *optimize(FGraphNode *weight,
									 FGraphNode *gradient) = 0;
};
struct Adam : public Optimizer {
		float epsilon;
		float learning_rate, b1, b2;
		Adam() = default;
		Adam(const Adam &) = delete;
		Adam(Adam &) = delete;
		Adam &operator=(const Adam &) = delete;
		Adam &operator=(Adam &) = delete;
		Adam(float learning_rate, float b1, float b2,
			 float epsilon = std::numeric_limits<float>::epsilon())
			: epsilon(epsilon), learning_rate(learning_rate), b1(b1), b2(b2) {}
		~Adam() {
			if (m) {
				m->reference_counter--;
				fFreeGraph(m);
			}
			if (v) {
				v->reference_counter--;
				fFreeGraph(v);
			}
		}
		FGraphNode *optimize(FGraphNode *weight, FGraphNode *gradient) override;

	private:
		FGraphNode *m = nullptr;
		FGraphNode *v = nullptr;
		size_t t = 1;
};
struct LossFunction {
		/**
		 * Calculates the loss between the actual output of the model and the
		 * expected output from the trainings data.
		 */
		virtual FGraphNode *calculate_loss(FGraphNode *actual,
										   FGraphNode *expected) = 0;
};
struct TrainingMetrics {
		/** if true a epoch has been trained, else it returns the
		 * metrics for a single batch and only some members are set.*/
		bool is_epoch;
		/** The average loss for the training dataset for the epoch (if
		 * `is_epoch` is false it is the loss of the single batch) */
		double training_loss;
		/** The average loss for the validation dataset for the epoch (not set
		 * if `is_epoch` is false) */
		double validation_loss;
		/** The combined time for the training dataset for the epoch (if
		 * `is_epoch` is false it is not set) */
		double training_time_ms;
		/** The time for the validation dataset for the epoch (if
		 * `is_epoch` is false it is not set) */
		double validation_time_ms;
		/** Average time for passing a batch through the model (if `is_epoch` is
		 * false, it is the time of the single batch)*/
		double avg_batch_time_ms;
		/** Average time for passing a batch through the model per layer (if
		 * `is_epoch` is false, it is the time of the single batch). Each layer
		 * is given with its name and its execution time.*/
		std::vector<std::pair<std::string, double>> avg_batch_time_per_layer_ms;
};
struct Trainer {
		DataLoader *data = nullptr;
		GraphModel *model = nullptr;
		Optimizer *optimizer = nullptr;
		LossFunction *loss = nullptr;
		size_t epochs;
		std::optional<double> early_stopping_error;
		/**
		 * Initializes the data of the Trainer.
		 * The `DataLoader`, `GraphModel` and `Optimizer` have to be maintained
		 * by whoever passed them and they have to live at least as long as the
		 * `Trainer`. The data for the training and validation will be taken
		 * from the `DataLoader`. The model `model` will be trained.
		 * The `opt` optimizer will be used to optimize the weights after each
		 * batch is passed through the model. The `loss` Loss function
		 * calculates the loss between the output of the model and the expected
		 * output from the labeled dataset.
		 */
		Trainer(GraphModel *model, DataLoader *dl, Optimizer *opt,
				LossFunction *loss)
			: model(model), data(dl), optimizer(opt), loss(loss) {}
		/**
		 * Initializes the model that should be trained by the Trainer.
		 * The `GraphModel` has to be maintained by whoever passed it and it has
		 * to live at least as long as the `Trainer`.
		 * The model `model` will be trained.
		 */
		Trainer(GraphModel *model) : model(model) {}
		Trainer() {};

		/**
		 * Enables the early stopping criterion for the following training runs.
		 * I.e. even if the minimum number of epochs is not reached, training
		 * will stop once the validation error reaches or is bellow the given
		 * minimum.
		 */
		void enable_early_stopping(double error) {
			early_stopping_error = error;
		}

		/**
		 * Sets the data of the Trainer.
		 * The `DataLoader` has to be maintained by whoever passed it and it has
		 * to live at least as long as the trainer. The data for the training
		 * and validation will be taken from the `DataLoader`.
		 */
		void set_data_loader(DataLoader *dl) { this->data = dl; }
		/**
		 * The `Optimizer` has to be maintained by whoever passed it and it has
		 * to live at least as long as the trainer.
		 * It will be used to optimize the weights after each
		 * batch is passed through the model.
		 */
		void set_optimizer(Optimizer *opt) { this->optimizer = opt; }

		/**
		 * The `LossFunction` has to be maintained by whoever passed it and it
		 * has to live at least as long as the trainer. It will be used to
		 * calculate the error of the model per batch for optimization.
		 */
		void set_loss(LossFunction *loss) { this->loss = loss; }
		/**
		 * Trains exactly one epoch, i.e., the complete dataset is passed
		 * through the model by splitting it into `batch_size` batches and
		 * passing them through the model. The weights of the model are
		 * optimized for each batch. If a validation dataset is available in the
		 * dataloader it is evaluated. This method returns informations (average
		 * loss, validation loss, total time, etc.) about the training.
		 *
		 * If a `TrainingReporter` is set, it reports the metrics per batch
		 */
		TrainingMetrics train_epoch();
		/**
		 * Trains the model for `epochs` number of epochs.
		 * The complete dataset is passed through the model per epoch
		 * (It is split into `batch_size` sized slices in the first dimension of
		 * the input data and each batch has to be passed through the model once
		 * per epoch). The weights of the model are optimized after each batch.
		 * Once all batches have been run for a epoch, the validation data is
		 * passed through the model and the error reported.
		 *
		 * Make sure the `DataLoader` and `Model` are valid for the call of this
		 * function.
		 */
		void train(size_t epochs);
};

#endif
