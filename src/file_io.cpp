/* Copyright 2022 David Schwarzbeck

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
  This file contains the implementation of IO functions for the C frontend
*/
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../flint.h"
#include "utils.hpp"
#include "libs/stb_image.hpp"
#include "libs/stb_image_write.hpp"

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
  data_size += total_size_node * typeSize(node->operation->data_type);
  // data type + dimensions + shape
  data_size += sizeof(FType) + sizeof(int) +
               node->operation->dimensions * sizeof(size_t);
  char *data = safe_mal<char>(data_size);
  // // conversion // //
  // magic number
  data[0] = (char)((MAGIC_NUMBER >> (3 * 8)) & 0xff);
  data[1] = (char)((MAGIC_NUMBER >> (2 * 8)) & 0xff);
  data[2] = (char)((MAGIC_NUMBER >> (1 * 8)) & 0xff);
  data[3] = (char)(MAGIC_NUMBER & 0xff);
  size_t index = 4;
  // type
  for (int i = sizeof(FType) - 1; i >= 0; i--)
    data[index++] = ((node->operation->data_type >> (i * 8)) & 0xff);
  // dimensions
  for (int i = sizeof(int) - 1; i >= 0; i--)
    data[index++] = ((node->operation->dimensions >> (i * 8)) & 0xff);
  // shape
  for (int i = 0; i < node->operation->dimensions; i++) {
    for (int j = sizeof(size_t) - 1; j >= 0; j--) {
      data[index++] = ((node->operation->shape[i] >> (j * 8)) & 0xff);
    }
  }
  // data
  memcpy(&data[index], node->result_data->data,
              total_size_node * typeSize(node->operation->data_type));
  if (bytes_written)
    *bytes_written =
        index + total_size_node * typeSize(node->operation->data_type);
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
  std::memcpy(res, &data[index], total_size * typeSize((FType)data_type));
  FGraphNode *node = fCreateGraph((void *)res, total_size, (FType)data_type,
                                  shape.data(), shape.size());
  free(res);
  return node;
}
