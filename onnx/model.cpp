#include "model.hpp"
#include "onnx.proto3.pb.h"
#include "onnx/layers/layers.hpp"
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
			x = new Convolve();
		} else if (node.op_type() == "Relu")
			x = new Relu();
		else if (node.op_type() == "BatchNormalization") {
			x = new BatchNorm();
		} else if (node.op_type() == "Add")
			x = new Add();
		else if (node.op_type() == "GlobalAveragePool") {
			x = new MaxPool(); // TODO use average pool
		} else if (node.op_type() == "Flatten")
			x = new Flatten();
		else if (node.op_type() == "Gemm")
			x = new Connected();
		else
			flogging(F_ERROR, "Unknown Operation " + node.op_type());
	}
}

int main() {
	using namespace std;
	ifstream input("test/resnet50-v1-12.onnx",
				   std::ios::ate |
					   std::ios::binary); // open file and move current position
										  // in file to the end

	streamsize size = input.tellg(); // get current position in file
	input.seekg(0, ios::beg);		 // move to start of file

	vector<char> buffer(size);
	input.read(buffer.data(), size); // read raw data

	onnx::ModelProto model;
	model.ParseFromArray(buffer.data(), size); // parse protobuf
	auto graph = model.graph();
	auto nodes = graph.node(); // topological order, first should be output
	std::cout << "nodes:\n";
	for (auto node : nodes) {
		std::cout << node.name() << ": " << node.op_type() << " {";
		for (auto input = node.input().begin(); input != node.input().end();
			 input++) {
			std::cout << *input;
			if (input != node.input().end() - 1)
				std::cout << ",";
		}
		std::cout << "} -> {";
		for (auto output = node.output().begin(); output != node.output().end();
			 output++) {
			std::cout << *output;
			if (output != node.output().end() - 1)
				std::cout << ",";
		}
		std::cout << "} [";
		for (int i = 0; i < node.attribute_size(); i++) {
			std::cout << node.attribute(i).name() << "(";
			if (node.attribute(i).type() ==
				onnx::AttributeProto_AttributeType_INTS) {
				for (int j = 0; j < node.attribute(i).ints().size(); j++) {
					std::cout << node.attribute(i).ints(j);
					if (j != node.attribute(i).ints().size() - 1)
						std::cout << ", ";
				}
			}
			std::cout << ")";
			if (i != node.attribute_size() - 1)
				std::cout << ", ";
		}
		std::cout << "]" << std::endl;
	}
	return 0;
}
