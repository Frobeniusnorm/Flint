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
void Convolve::forward() {
#ifdef FLINT_DEBUG
	if ((incoming.size() != 2 && incoming.size() != 3) ||
		incoming[0]->output.size() != 1 || incoming[1]->output.size() != 1 ||
		(incoming.size() == 3 && incoming[2]->output.size() != 1)) {
		flogging(F_ERROR, "Convolve expects an image and a kernel as "
						  "parameters and optionally a bias");
	}
#endif
	FGraphNode *weight = incoming[1]->output[0];
	FGraphNode *image = incoming[0]->output[0];
	// switch channels and width
	int transpositions1[image->operation.dimensions];
	for (int i = 0; i < image->operation.dimensions; i++)
		transpositions1[i] = i;
	transpositions1[1] = image->operation.dimensions - 1;
	transpositions1[image->operation.dimensions - 1] = 1;
	image = ftranspose(image, transpositions1);
	int transpositions2[weight->operation.dimensions];
	for (int i = 0; i < weight->operation.dimensions; i++)
		transpositions2[i] = i;
	transpositions2[1] = weight->operation.dimensions - 1;
	transpositions2[weight->operation.dimensions - 1] = 1;
	weight = ftranspose(weight, transpositions2);
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
	// switch height and width
	if (steps.size() > 2) {
		const unsigned int tmp = steps[1];
		steps[1] = steps[2];
		steps[2] = tmp;
	}
	// adapt image with padding
	vector<size_t> padded_shape(image->operation.dimensions);
	vector<size_t> inclusion_index(image->operation.dimensions, 0);
	const size_t spatial_dims = image->operation.dimensions - 2;
	auto padding_for = [&](size_t spatial_idx,
						   bool end_padding) -> unsigned int {
		if (padding.empty())
			return 0;
		if (padding.size() >= spatial_dims * 2)
			return padding[spatial_idx + (end_padding ? spatial_dims : 0)];
		if (padding.size() >= spatial_dims)
			return padding[spatial_idx];
		flogging(F_ERROR, "Invalid padding size for convolution layer.");
		return 0;
	};
	for (int i = 0; i < padded_shape.size(); i++) {
		padded_shape[i] = image->operation.shape[i];
		if (padding.size() != 0 && i > 0 && i < padded_shape.size() - 1) {
			inclusion_index[i] = padding_for(i - 1, false);
			padded_shape[i] +=
				padding_for(i - 1, false) + padding_for(i - 1, true);
		}
	}
	image = fextend(image, padded_shape.data(), inclusion_index.data());
	// do the convolution
	output[0] = fconvolve(image, eweight, steps.data());
	if (bias)
		output[0] = fadd(output[0], bias);
	// switch channels back to front
	output[0] = ftranspose(output[0], transpositions1);
}

void MaxPool::forward() {
#ifdef FLINT_DEBUG
	if (incoming.size() != 1 || incoming[0]->output.size() != 1)
		flogging(F_ERROR, "MaxPool expects an image as inputs");
#endif
	using namespace std;
	vector<unsigned int> steps(incoming[0]->output[0]->operation.dimensions);
	if (steps.size() < 2 || stride.size() + 2 < steps.size())
		flogging(F_ERROR, "Invalid stride size for max pooling layer.");
	steps[0] = 1;
	steps[1] = 1; // don't touch channels
	for (int i = 2; i < steps.size(); i++)
		steps[i] = stride[i - 2];
	// adapt image with padding
	FGraphNode *image = incoming[0]->output[0];
	vector<size_t> padded_shape(image->operation.dimensions);
	vector<size_t> inclusion_index(image->operation.dimensions, 0);
	const size_t spatial_dims = image->operation.dimensions - 2;
	auto padding_for = [&](size_t spatial_idx,
						   bool end_padding) -> unsigned int {
		if (padding.empty())
			return 0;
		if (padding.size() >= spatial_dims * 2)
			return padding[spatial_idx + (end_padding ? spatial_dims : 0)];
		if (padding.size() >= spatial_dims)
			return padding[spatial_idx];
		flogging(F_ERROR, "Invalid padding size for max pooling layer.");
		return 0;
	};
	for (int i = 0; i < padded_shape.size(); i++) {
		padded_shape[i] = image->operation.shape[i];
		if (padding.size() != 0 && i > 1) {
			inclusion_index[i] = padding_for(i - 2, false);
			padded_shape[i] +=
				padding_for(i - 2, false) + padding_for(i - 2, true);
		}
	}
	image = fextend(image, padded_shape.data(), inclusion_index.data());
	size_t windows[kernel_shape.size() + 2];
	for (int i = 2; i < kernel_shape.size() + 2; i++)
		windows[i] = kernel_shape[i - 2];
	windows[0] = 1;
	windows[1] = 1; // don't touch the channels
	// do the pooling
	output[0] = fpooling_max(fexpand(image, image->operation.dimensions, 1),
							 windows, steps.data());
}
void AvgPool::forward() {
#ifdef FLINT_DEBUG
	if (incoming.size() != 1 || incoming[0]->output.size() != 1)
		flogging(F_ERROR, "AvgPool expects an image as inputs");
#endif
	using namespace std;
	vector<unsigned int> steps(stride.size() + 1);
	steps[0] = 1;
	for (int i = 1; i < steps.size(); i++)
		steps[i] = stride[i - 1];
	// adapt image with padding
	FGraphNode *image = incoming[0]->output[0];
	vector<size_t> padded_shape(image->operation.dimensions);
	vector<size_t> inclusion_index(image->operation.dimensions, 0);
	const size_t spatial_dims = image->operation.dimensions - 2;
	auto padding_for = [&](size_t spatial_idx,
						   bool end_padding) -> unsigned int {
		if (padding.empty())
			return 0;
		if (padding.size() >= spatial_dims * 2)
			return padding[spatial_idx + (end_padding ? spatial_dims : 0)];
		if (padding.size() >= spatial_dims)
			return padding[spatial_idx];
		flogging(F_ERROR, "Invalid padding size for avg pooling layer.");
		return 0;
	};
	for (int i = 0; i < padded_shape.size(); i++) {
		padded_shape[i] = image->operation.shape[i];
		if (padding.size() != 0 && i > 0 && i < padded_shape.size() - 1) {
			inclusion_index[i] = padding_for(i - 1, false);
			padded_shape[i] +=
				padding_for(i - 1, false) + padding_for(i - 1, true);
		}
	}
	image = fextend(image, padded_shape.data(), inclusion_index.data());
	size_t windows[kernel_shape.size() + 1];
	size_t window_total = 1;
	for (int i = 1; i < kernel_shape.size() + 1; i++) {
		windows[i] = kernel_shape[i - 1];
		window_total *= windows[i];
	}
	windows[0] = 1;
	// do the pooling
	output[0] = fexpand(
		fdiv_ci(fpooling_sum(image, windows, steps.data()), window_total),
		image->operation.dimensions - 1, 1);
}
void GlobalAvgPool::forward() {
#ifdef FLINT_DEBUG
	if (incoming.size() != 1 || incoming[0]->output.size() != 1)
		flogging(F_ERROR, "AvgPool expects an image as inputs");
#endif
	using namespace std;
	FGraphNode *image = incoming[0]->output[0];
	// only first and last dimension are kept
	const int dims = image->operation.dimensions;
	while (image->operation.dimensions > 2) {
		const size_t shape_size = image->operation.shape[2];
		image = fdiv_ci(freduce_sum(image, 2), shape_size);
	}
	// expand to fit original rank
	vector<size_t> rank_shape(dims, 1);
	rank_shape[0] = image->operation.shape[0];
	rank_shape[1] = image->operation.shape[1];
	output[0] = freshape(image, rank_shape.data(), dims);
}
