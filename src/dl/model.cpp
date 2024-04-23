#include "model.hpp"
#include "layers.hpp"
#include "onnx.proto3.pb.h"
#include <fstream>
#include <iostream>
#include <list>
#include <set>
GraphModel *GraphModel::load_model(std::string path) {
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
	vector<Variable *> weights;
	// first we parse the weights
	for (auto &init : graph.initializer()) {
		Variable *var = new Variable();
		switch (init.data_type()) {
		case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT: {
			var->node = fCreateGraph(
				init.float_data().data(), init.float_data_size(), F_FLOAT32,
				(const unsigned long *)init.dims().data(), init.dims().size());
			break;
		}
		case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE: {
			var->node = fCreateGraph(
				init.double_data().data(), init.double_data_size(), F_FLOAT64,
				(const unsigned long *)init.dims().data(), init.dims().size());
			break;
		}
		case onnx::TensorProto_DataType::TensorProto_DataType_INT32: {
			var->node = fCreateGraph(
				init.int32_data().data(), init.int32_data_size(), F_INT32,
				(const unsigned long *)init.dims().data(), init.dims().size());
			break;
		}
		case onnx::TensorProto_DataType::TensorProto_DataType_INT64: {
			var->node = fCreateGraph(
				init.int64_data().data(), init.int64_data_size(), F_INT64,
				(const unsigned long *)init.dims().data(), init.dims().size());
			break;
		}
		default:
			flogging(F_ERROR, "Unknown type: " + to_string(init.data_type()));
		}
		layers.insert({init.name(), var});
		weights.push_back(var);
	}
	// now we process the layers
	InputNode *in = new InputNode();
	layers.insert({"data", in});
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
						stride[j] = attr.ints()[j];
				}
			}
			x = new Convolve(stride, padding);
		} else if (node.op_type() == "Relu") {
			x = new Relu();
		} else if (node.op_type() == "BatchNormalization") {
			x = new BatchNorm();
		} else if (node.op_type() == "Add") {
			x = new Add();
		} else if (node.op_type() == "GlobalAveragePool") {
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
						stride[j] = attr.ints()[j];
				} else if (attr.name() == "kernel_shape") {
					kernel_shape = vector<size_t>(attr.ints_size());
					for (int j = 0; j < attr.ints_size(); j++)
						kernel_shape[j] = attr.ints()[j];
				}
			}
			x = new MaxPool(kernel_shape, stride, padding);
		} else if (node.op_type() == "Flatten") {
			x = new Flatten();
		} else if (node.op_type() == "Gemm") {
			Connected *conn = new Connected();
			for (int i = 0; i < node.attribute_size(); i++) {
				auto &attr = node.attribute(i);
				if (attr.name() == "transA") {
					if (attr.i() == 1) {
						conn->transposeA = true;
					}
				} else if (attr.name() == "transB") {
					if (attr.i() == 1) {
						conn->transposeB = true;
					}
				}
			}
			x = conn;
		} else
			flogging(F_ERROR, "Unknown Operation " + node.op_type());
		x->name = node.name();
		layers.insert({node.name(), x});
		for (int i = 0; i < node.input_size(); i++) {
			const auto &in = node.input(i);
			if (layers.contains(in)) {
				LayerGraph *conn = layers[in];
				x->incoming.push_back(conn);
				conn->outgoing.push_back(x);
			} else
				flogging(F_WARNING, "Unknown input: " + in);
		}
		if (in->outgoing.empty()) {
			in->outgoing.push_back(x);
		}
	}
	GraphModel *res = new GraphModel();
	res->input = in;
	res->output = layers[graph.output(0).name()];
	res->weights = weights;
	return res;
}

FGraphNode *GraphModel::operator()(FGraphNode *in) {
	return this->operator()(std::vector{in})[0];
}
std::vector<FGraphNode *> GraphModel::operator()(std::vector<FGraphNode *> in) {
	input->nodes.clear();
	input->nodes.insert(input->nodes.end(), in.begin(), in.end());
	using namespace std;
	list<LayerGraph *> todo;
	set<LayerGraph *> visited;
	//	set<FGraphNode *> holding = {in};
	todo.insert(todo.begin(), input->outgoing.begin(), input->outgoing.end());
	visited.insert(input);
	// todo.push_back(nullptr); // as a marker that the holding reference
	//							 // can be removed
	for (Variable *v : weights) {
		v->forward();
		visited.insert(v);
	}
	while (!todo.empty()) {
		LayerGraph *curr = todo.front();
		todo.pop_front();
		// check if previous layers have been visited
		bool all_run = true;
		for (LayerGraph *in : curr->incoming) {
			if (!visited.contains(in)) {
				all_run = false;
				break;
			}
		}
		if (!all_run) {
			todo.push_back(curr);
			continue;
		}
		if (curr) {
			flogging(F_INFO, "Layer " + curr->name);
			curr->forward();
			visited.insert(curr);
			if (curr == output) {
				for (FGraphNode *out : curr->output) {
					out->reference_counter++;
				}
			}
			for (FGraphNode *out : curr->output) {
				std::cout << "[";
				for (size_t i = 0; i < out->operation.dimensions; i++) {
					std::cout << out->operation.shape[i];
					if (i != out->operation.dimensions - 1)
						std::cout << ", ";
				}
				std::cout << "]\n";
			}
			// add the children bfs
			for (LayerGraph *layer : curr->outgoing) {
				auto ex = std::find(todo.begin(), todo.end(), layer);
				if (ex != todo.end())
					todo.erase(ex);
				todo.push_back(layer);
			}
		}
		//  else {
		//		flogging(F_INFO, "Clearing");
		//		for (FGraphNode *h : holding) {
		//			h->reference_counter--;
		//			// maybe it is already no longer needed
		//			fFreeGraph(h);
		//		}
		//		holding.clear();
		//		if (!todo.empty() && todo.front() != nullptr)
		//			todo.push_back(nullptr);
		//	}
		for (FGraphNode *out : output->output) {
			out->reference_counter--;
		}
	}
	return output->output;
}
GraphModel *GraphModel::sequential(std::vector<LayerGraph *> list) {
	GraphModel *model = new GraphModel();
	model->input = new InputNode(); // TODO
	LayerGraph *prev = model->input;
	for (LayerGraph *node : list) {
		prev->outgoing.push_back(node);
		for (LayerGraph *pw : prev->outgoing) {
			if (Variable *w = dynamic_cast<Variable *>(pw))
				model->weights.push_back(w);
		}
		node->incoming.insert(node->incoming.begin(), prev);
	}
	model->output = prev;
	return model;
}
GraphModel *GraphModel::from_output(LayerGraph *output) {
	GraphModel *model = new GraphModel();
	model->output = output;
	// iterate and find all inputs and weights
	using namespace std;
	list<LayerGraph *> todo;
	set<LayerGraph *> visited;
	todo.push_back(output);
	while (!todo.empty()) {
		LayerGraph *curr = todo.front();
		todo.pop_front();
		if (Variable *v = dynamic_cast<Variable *>(curr)) {
			model->weights.push_back(v);
		}
		if (InputNode *i = dynamic_cast<InputNode *>(curr)) {
			model->input = i; // TODO
		}
		for (LayerGraph *n : curr->incoming) {
			if (visited.insert(n).second) {
				todo.push_back(n);
			}
		}
	}
}
