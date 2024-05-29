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
};
/**
 * Interface to optimize variables
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
		FGraphNode *optimize(FGraphNode *weight, FGraphNode *gradient) override;
};

struct Trainer {
		DataLoader *data = nullptr;
		GraphModel *model = nullptr;
		Optimizer *optimizer = nullptr;
		size_t epochs;
		size_t batch_size;
		std::optional<double> early_stopping_error;
		/**
		 * Initializes the data of the Trainer.
		 * The `DataLoader`, `GraphModel` and `Optimizer` have to be maintained
		 * by whoever passed them and they have to live at least as long as the
		 * `Trainer`. The data for the training and validation will be taken
		 * from the `DataLoader`. The model `model` will be trained.
		 * The `opt` optimizer will be used to optimize the weights after each
		 * batch is passed through the model.
		 */
		Trainer(GraphModel *model, DataLoader *dl, Optimizer* opt) : model(model), data(dl), optimizer(opt) {}
		/**
		 * Initializes the model that should be trained by the Trainer.
		 * The `GraphModel` has to be maintained by whoever passed it and it has
		 * to live at least as long as the `Trainer`.
		 * The model `model` will be trained.
		 */
		Trainer(GraphModel *model) : model(model) {}
		Trainer(){};

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
		void train(size_t epochs, size_t batch_size) {}
};

#endif
