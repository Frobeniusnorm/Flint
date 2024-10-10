#include "../trainer.hpp"
#include "flint.h"
#include <fstream>
int reverseInt(int i) {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
static FGraphNode *load_idx_images(const std::string path) {
	using namespace std;
	errno = 0;
	ifstream file(path);
	if (file.is_open()) {
		int magic_number = 0;
		int no = 0;
		int h = 0;
		int w = 0;
		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		file.read((char *)&no, sizeof(no));
		no = reverseInt(no);
		file.read((char *)&h, sizeof(h));
		h = reverseInt(h);
		file.read((char *)&w, sizeof(w));
		w = reverseInt(w);
		std::vector<float> data(no * h * w);
		for (int i = 0; i < no; i++) {
			for (int j = 0; j < h; j++) {
				for (int k = 0; k < w; k++) {
					unsigned char value;
					file.read((char *)&value, 1);
					data[i * h * w + j * w + k] = (float)value / 255.0;
				}
			}
		}
		std::array<size_t, 3> shape{(size_t)no, (size_t)h, (size_t)w};
		return fCreateGraph(data.data(), no * h * w, F_FLOAT32, shape.data(),
							shape.size());
	} else
		throw std::runtime_error("Could not load file " + path);
}
static FGraphNode *load_idx_labels(const std::string path) {
	using namespace std;
	errno = 0;
	ifstream file(path);
	if (file.is_open()) {
		int magic_number = 0;
		int no = 0;
		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		file.read((char *)&no, sizeof(no));
		no = reverseInt(no);
		std::vector<int> data(no * 10);
		for (int i = 0; i < no; i++) {
			unsigned char value;
			file.read((char *)&value, 1);
			for (int j = 0; j < 10; j++) {
				data[i * 10 + j] = value == j ? 1 : 0;
			}
		}
		std::array<size_t, 2> shape{(size_t)no, 10};
		return fCreateGraph(data.data(), no * 10, F_INT32, shape.data(), 2);
	} else
		throw std::runtime_error("Could not load file " + path);
}
void IDXFormatLoader::prefetch_data() {
	training_data = load_idx_images(train_images_path);
	training_labels = load_idx_labels(train_labels_path);
	if (test_images_path != "" && test_labels_path != "") {
		test_data = load_idx_images(train_images_path);
		test_labels = load_idx_labels(train_labels_path);
	}
	if (validation_percentage > 0.0) {
		// split validation set
		const size_t validation_size =
			(size_t)(training_data->operation.shape[0] * validation_percentage);
		using namespace std;
		vector<size_t> indices(training_data->operation.shape[0]);
		for (size_t i = 0; i < indices.size(); i++)
			indices[i] = i;
		// shuffle
		for (size_t i = 0; i < indices.size(); i++) {
			const size_t a = rand() % indices.size();
			const size_t b = rand() % indices.size();
			const size_t tmp = indices[a];
			indices[a] = indices[b];
			indices[b] = tmp;
		}
		vector<size_t> index_train(training_data->operation.shape[0] -
								   validation_size);
		size_t i = 0;
		for (; i < index_train.size(); i++)
			index_train[i] = indices[i];
		vector<size_t> index_validate(validation_size);
		for (; i < indices.size(); i++)
			index_validate[i - index_train.size()] = indices[i];
		// slice from training data
		validation_data =
			findex(training_data,
				   fCreateGraph(index_validate.data(), index_validate.size(),
								F_FLOAT64, index_validate.data(), 1));
		validation_data->reference_counter++;
		validation_labels =
			findex(training_labels,
				   fCreateGraph(index_validate.data(), index_validate.size(),
								F_FLOAT64, index_validate.data(), 1));
		validation_labels->reference_counter++;
		training_data = findex(
			training_data, fCreateGraph(index_train.data(), index_train.size(),
										F_FLOAT64, index_train.data(), 1));
		training_data->reference_counter++;
		training_labels =
			findex(training_labels,
				   fCreateGraph(index_train.data(), index_train.size(),
								F_FLOAT64, index_train.data(), 1));
		training_labels->reference_counter++;
	}
}
std::pair<std::vector<FGraphNode *>, std::vector<FGraphNode *>>
IDXFormatLoader::next_batch() {
	if (batch_index * batch_size >= training_labels->operation.shape[0]) {
		batch_index = 0;
		batch_indices->reference_counter--;
		batch_indices = fpermutate(batch_indices, 0);
	}
	if (!batch_indices) {
		batch_indices =
			fpermutate(farange(training_labels->operation.shape, 1, 0), 0);
		batch_indices->reference_counter++;
	}
	const long cur_batch_index = batch_index;
	const long new_batch_index = ++batch_index;
	FGraphNode *actual_indices =
		fslice(batch_indices, &cur_batch_index, &new_batch_index);
	actual_indices->reference_counter++;
	FGraphNode *sel_labels = findex(training_labels, actual_indices);
	FGraphNode *sel_images = findex(training_data, actual_indices);
	actual_indices->reference_counter--;
	return {{sel_labels}, {sel_images}};
}
std::pair<std::vector<FGraphNode *>, std::vector<FGraphNode *>>
IDXFormatLoader::validation_batch() {
	return {{validation_labels}, {validation_data}};
}
std::pair<std::vector<FGraphNode *>, std::vector<FGraphNode *>>
IDXFormatLoader::testing_data() {
	return {{test_labels}, {test_data}};
}

size_t IDXFormatLoader::remaining_for_epoch() {
	const size_t total_batches =
		(training_labels->operation.shape[0] - 1) / batch_size;
	return total_batches - batch_index;
}
