#include "../layers.hpp"
#include "flint.h"
#include <iostream>
// // //
// static values
// // //
int Convolve::conv_no = 0;
int MaxPool::mpool_no = 0;
int AvgPool::apool_no = 0;
int GlobalAvgPool::gapool_no = 0;
// // //
// implementations
// // //
/** Pads the spatial dimensions of an image [batch, spatial..., channels] with
 * zeros. `padding` holds one value per spatial dimension, or one for the
 * beginning and one for the end of each. */
static FGraphNode *pad_spatial(FGraphNode *image,
							   const std::vector<unsigned int> &padding,
							   const std::string &layer) {
	using namespace std;
	const int dims = image->operation.dimensions;
	const size_t spatial_dims = dims - 2;
	auto padding_for = [&](size_t spatial_idx,
						   bool end_padding) -> unsigned int {
		if (padding.empty())
			return 0;
		if (padding.size() >= spatial_dims * 2)
			return padding[spatial_idx + (end_padding ? spatial_dims : 0)];
		if (padding.size() >= spatial_dims)
			return padding[spatial_idx];
		flogging(F_ERROR, "Invalid padding size for " + layer + " layer.");
		return 0;
	};
	vector<size_t> padded_shape(dims);
	vector<size_t> inclusion_index(dims, 0);
	bool padded = false;
	for (int i = 0; i < dims; i++) {
		padded_shape[i] = image->operation.shape[i];
		if (i > 0 && i < dims - 1) {
			inclusion_index[i] = padding_for(i - 1, false);
			padded_shape[i] += inclusion_index[i] + padding_for(i - 1, true);
			padded |= padded_shape[i] != image->operation.shape[i];
		}
	}
	return padded ? fextend(image, padded_shape.data(), inclusion_index.data())
				  : image;
}
void Convolve::forward() {
#ifdef FLINT_DEBUG
	if ((incoming.size() != 2 && incoming.size() != 3) ||
		incoming[0]->output.size() != 1 || incoming[1]->output.size() != 1 ||
		(incoming.size() == 3 && incoming[2]->output.size() != 1)) {
		flogging(F_ERROR, "Convolve expects an image and a kernel as "
						  "parameters and optionally a bias");
	}
#endif
	// images are [batch, spatial..., channels] and kernels [filters,
	// spatial..., channels], which is already what fconvolve expects
	FGraphNode *weight = incoming[1]->output[0];
	FGraphNode *image = incoming[0]->output[0];
	FGraphNode *bias = incoming.size() == 3 ? incoming[2]->output[0] : nullptr;
	// expand kernel s.t. it matches the batch size
	FGraphNode *eweight = fexpand(weight, 1, 1);
	using namespace std;
	vector<unsigned int> steps(stride.size() + 1);
	if (steps.size() != image->operation.dimensions - 1)
		flogging(F_ERROR, "Invalid stride size for convolution layer.");
	steps[0] = 1;
	for (int i = 1; i < steps.size(); i++)
		steps[i] = stride[i - 1];
	image = pad_spatial(image, padding, "convolution");
	// do the convolution
	output[0] = fconvolve(image, eweight, steps.data());
	if (bias)
		output[0] = fadd(output[0], bias);
}

void MaxPool::forward() {
#ifdef FLINT_DEBUG
	if (incoming.size() != 1 || incoming[0]->output.size() != 1)
		flogging(F_ERROR, "MaxPool expects an image as inputs");
#endif
	using namespace std;
	FGraphNode *image = incoming[0]->output[0];
	const int dims = image->operation.dimensions;
	if (dims < 3 || stride.size() + 2 < dims)
		flogging(F_ERROR, "Invalid stride size for max pooling layer.");
	// the image is [batch, spatial..., channels], only spatial is pooled
	vector<unsigned int> steps(dims, 1);
	vector<size_t> windows(dims, 1);
	for (int i = 1; i < dims - 1; i++) {
		steps[i] = stride[i - 1];
		windows[i] = kernel_shape[i - 1];
	}
	// pooling reduces the last dimension completely, so the channels get a
	// trailing one to keep them
	FGraphNode *padded = pad_spatial(image, padding, "max pooling");
	output[0] =
		fpooling_max(fexpand(padded, dims, 1), windows.data(), steps.data());
}
void AvgPool::forward() {
#ifdef FLINT_DEBUG
	if (incoming.size() != 1 || incoming[0]->output.size() != 1)
		flogging(F_ERROR, "AvgPool expects an image as inputs");
#endif
	using namespace std;
	FGraphNode *image = incoming[0]->output[0];
	const int dims = image->operation.dimensions;
	if (dims < 3 || stride.size() + 2 < dims)
		flogging(F_ERROR, "Invalid stride size for avg pooling layer.");
	// the image is [batch, spatial..., channels], only spatial is pooled
	vector<unsigned int> steps(dims, 1);
	vector<size_t> windows(dims, 1);
	size_t window_total = 1;
	for (int i = 1; i < dims - 1; i++) {
		steps[i] = stride[i - 1];
		windows[i] = kernel_shape[i - 1];
		window_total *= windows[i];
	}
	// pooling reduces the last dimension completely, so the channels get a
	// trailing one to keep them
	FGraphNode *padded = pad_spatial(image, padding, "avg pooling");
	output[0] = fdiv_ci(
		fpooling_sum(fexpand(padded, dims, 1), windows.data(), steps.data()),
		window_total);
}
void GlobalAvgPool::forward() {
#ifdef FLINT_DEBUG
	if (incoming.size() != 1 || incoming[0]->output.size() != 1)
		flogging(F_ERROR, "AvgPool expects an image as inputs");
#endif
	using namespace std;
	FGraphNode *image = incoming[0]->output[0];
	// only batch and channels (first and last dimension) are kept
	const int dims = image->operation.dimensions;
	while (image->operation.dimensions > 2) {
		const size_t shape_size = image->operation.shape[1];
		image = fdiv_ci(freduce_sum(image, 1), shape_size);
	}
	// expand to fit original rank
	vector<size_t> rank_shape(dims, 1);
	rank_shape[0] = image->operation.shape[0];
	rank_shape[dims - 1] = image->operation.shape[1];
	output[0] = freshape(image, rank_shape.data(), dims);
}
