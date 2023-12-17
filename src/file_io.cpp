/* Copyright 2023 David Schwarzbeck
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../flint.h"
#include "errors.hpp"
#include "libs/stb_image.hpp"
#include "libs/stb_image_write.hpp"
#include "utils.hpp"
/* Dataformat
 * Magic Number
 * data_type (4 bytes)
 * dimensions (4 bytes)
 * list of sizes per dimension (each 4 bytes)
 * data
 */
#define MAGIC_NUMBER 0x75321
char *fserialize(FGraphNode *node, size_t *bytes_written) {
	if (!node->result_data)
		fExecuteGraph(node);
	if (!node->result_data->data)
		fSyncMemory(node);
	size_t total_size_node = node->result_data->num_entries;
	// header
	size_t data_size = 4;
	// pure data
	data_size += total_size_node * typeSize(node->operation.data_type);
	// data type + dimensions + shape
	data_size += sizeof(FType) + sizeof(int) +
				 node->operation.dimensions * sizeof(size_t);
	char *data = safe_mal<char>(data_size);
	if (!data)
		return nullptr;
	// // conversion // //
	// magic number
	data[0] = (char)((MAGIC_NUMBER >> (3 * 8)) & 0xff);
	data[1] = (char)((MAGIC_NUMBER >> (2 * 8)) & 0xff);
	data[2] = (char)((MAGIC_NUMBER >> (1 * 8)) & 0xff);
	data[3] = (char)(MAGIC_NUMBER & 0xff);
	size_t index = 4;
	// type
	for (int i = sizeof(FType) - 1; i >= 0; i--)
		data[index++] = ((node->operation.data_type >> (i * 8)) & 0xff);
	// dimensions
	for (int i = sizeof(int) - 1; i >= 0; i--)
		data[index++] = ((node->operation.dimensions >> (i * 8)) & 0xff);
	// shape
	for (int i = 0; i < node->operation.dimensions; i++) {
		for (int j = sizeof(size_t) - 1; j >= 0; j--) {
			data[index++] = ((node->operation.shape[i] >> (j * 8)) & 0xff);
		}
	}
	// data
	memcpy(&data[index], node->result_data->data,
		   total_size_node * typeSize(node->operation.data_type));
	if (bytes_written)
		*bytes_written =
			index + total_size_node * typeSize(node->operation.data_type);
	return data;
}

FGraphNode *fdeserialize(char *data) {
	int m = (data[0] << (3 * 8)) | (data[1] << (2 * 8)) | (data[2] << (1 * 8)) |
			(data[3]);
	if (m != MAGIC_NUMBER) {
		flogging(F_WARNING, "Node could not be constructed from binary data!");
		return nullptr;
	}
	size_t index = 4;
	int data_type = 0;
	for (int i = sizeof(FType) - 1; i >= 0; i--)
		data_type |= data[index++] << (i * 8);
	int dimensions = 0;
	for (int i = sizeof(int) - 1; i >= 0; i--)
		dimensions |= data[index++] << (i * 8);
	std::vector<size_t> shape(dimensions, 0);
	size_t total_size = 1;
	for (int i = 0; i < dimensions; i++) {
		for (int j = sizeof(size_t) - 1; j >= 0; j--) {
			shape[i] |= data[index++] << (j * 8);
		}
		total_size *= shape[i];
	}
	char *res = safe_mal<char>(total_size * typeSize((FType)data_type));
	if (!res)
		return nullptr;
	memcpy(res, &data[index], total_size * typeSize((FType)data_type));
	FGraphNode *node = fCreateGraph((void *)res, total_size, (FType)data_type,
									shape.data(), shape.size());
	free(res);
	return node;
}
FGraphNode *fload_image(const char *path) {
	int w, h, c;
	unsigned char *vals = stbi_load(path, &w, &h, &c, 0);
	if (!vals) {
		setErrorType(IO_ERROR);
		flogging(F_ERROR, "Could not load image!");
		return nullptr;
	}
	float *fvals = safe_mal<float>(w * h * c);
	if (!fvals)
		return nullptr;
	for (int i = 0; i < w * h * c; i++)
		fvals[i] = vals[i] / 255.f;
	size_t shape[3] = {(size_t)h, (size_t)w, (size_t)c};
	FGraphNode *node = fCreateGraph(fvals, w * h * c, F_FLOAT32, &shape[0], 3);
	stbi_image_free(vals);
	free(fvals);
	return node;
}
FErrorType fstore_image(FGraphNode *node, const char *path,
						FImageFormat format) {
	FGraphNode *orig = node;
	if (node->operation.data_type != F_FLOAT32 ||
		node->operation.dimensions != 3) {
		FErrorType error = node->operation.data_type != F_FLOAT32
							   ? WRONG_TYPE
							   : ILLEGAL_DIMENSIONALITY;
		setErrorType(error);
		flogging(
			F_ERROR,
			"Invalid image data for fstore_image: image nodes are expected to "
			"have 3 dimensions and to be of the float data type!");
		return error;
	}
	int h = node->operation.shape[0], w = node->operation.shape[1],
		c = node->operation.shape[2];
	node = fmin_ci(fmax_ci(fconvert(fmul(node, 255.0f), F_INT32), 0), 255);
	if (!node)
		return fErrorType();
	node = fCalculateResult(node);
	if (!node)
		return fErrorType();
	char *data = nullptr;
	data = safe_mal<char>(node->result_data->num_entries);
	if (!data)
		return OUT_OF_MEMORY;
	for (size_t i = 0; i < node->result_data->num_entries; i++)
		data[i] = (char)((int *)node->result_data->data)[i];
	switch (format) {
	case F_PNG:
		if (!stbi_write_png(path, w, h, c, data, 0)) {
			setErrorType(IO_ERROR);
			flogging(F_ERROR, "Could not write image!");
			return IO_ERROR;
		}
		break;
	case F_JPEG:
		if (!stbi_write_jpg(path, w, h, c, data, 70)) {
			setErrorType(IO_ERROR);
			flogging(F_ERROR, "Could not write image!");
			return IO_ERROR;
		}
	case F_BMP:
		if (!stbi_write_bmp(path, w, h, c, data)) {
			setErrorType(IO_ERROR);
			flogging(F_ERROR, "Could not write image!");
			return IO_ERROR;
		}
	}
	if (data)
		free(data);
	orig->reference_counter++;
	fFreeGraph(node);
	orig->reference_counter--;
	return NO_ERROR;
}
