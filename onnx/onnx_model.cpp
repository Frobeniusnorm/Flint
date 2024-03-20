#include "onnx_model.hpp"
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
    std::cout << "}\n";
	}
	return 0;
}