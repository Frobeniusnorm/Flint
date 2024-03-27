#include "model.hpp"
#include "layers/layers.hpp"
#include "onnx.proto3.pb.h"
#include <fstream>
#include <iostream>

GraphModel GraphModel::load_model(std::string path) {
	using namespace std;
	ifstream input(path, std::ios::ate | std::ios::binary);
	streamsize size = input.tellg();
	input.seekg(0, ios::beg);

	vector<char> buffer(size);
	input.read(buffer.data(), size);

	onnx::ModelProto model;
	model.ParseFromArray(buffer.data(), size);
	auto graph = model.graph();
	auto nodes = graph.node();
	unordered_map<string, LayerGraph *> layers;
	InputNode *in;
	for (auto node : nodes) {
		LayerGraph *x;
		if (node.op_type() == "Conv") {
			vector<unsigned int> stride, padding;
			for (int i = 0; i < node.attribute_size(); i++) {
				auto &attr = node.attribute(i);
				if (attr.name() == "pads") {
					padding = vector<unsigned int>(attr.ints_size());
					for (int j = 0; j < attr.ints_size(); j++)
						padding[j] = attr.ints()[j];
				} else if (attr.name() == "strides") {
					stride = vector<unsigned int>(attr.ints_size());
					for (int j = 0; j < attr.ints_size(); j++)
						padding[j] = attr.ints()[j];
				}
			}
			x = new Convolve(stride, padding);
		} else if (node.op_type() == "Relu")
			x = new Relu();
		else if (node.op_type() == "BatchNormalization") {
			x = new BatchNorm();
		} else if (node.op_type() == "Add")
			x = new Add();
		else if (node.op_type() == "GlobalAveragePool") {
			x = new GlobalAvgPool();
		} else if (node.op_type() == "MaxPool") {
			vector<unsigned int> stride, padding;
			vector<size_t> kernel_shape;
			for (int i = 0; i < node.attribute_size(); i++) {
				auto &attr = node.attribute(i);
				if (attr.name() == "pads") {
					padding = vector<unsigned int>(attr.ints_size());
					for (int j = 0; j < attr.ints_size(); j++)
						padding[j] = attr.ints()[j];
				} else if (attr.name() == "strides") {
					stride = vector<unsigned int>(attr.ints_size());
					for (int j = 0; j < attr.ints_size(); j++)
						padding[j] = attr.ints()[j];
				} else if (attr.name() == "kernel_shape") {
					kernel_shape = vector<size_t>(attr.ints_size());
					for (int j = 0; j < attr.ints_size(); j++)
						kernel_shape[j] = attr.ints()[j];
				}
			}
			x = new MaxPool(kernel_shape, stride, padding);
		} else if (node.op_type() == "Flatten")
			x = new Flatten();
		else if (node.op_type() == "Gemm")
			x = new Connected();
		else
			flogging(F_ERROR, "Unknown Operation " + node.op_type());
	}
}
