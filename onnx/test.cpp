#include "onnx.proto3.pb.h"
#include <fstream>
#include <iostream>
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
