#include "model.hpp"
#include "flint.h"
#include "layers.hpp"
#include "onnx.proto3.pb.h"
#include <fstream>
#include <iostream>
#include <list>
#include <set>
// // //
// static data
// // //
int Variable::variable_no = 0;
int InputNode::data_no = 0;
// // //
// implementations
// // //
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
	// parse the weights
	for (auto &init : graph.initializer()) {
		Variable *var;
		switch (init.data_type()) {
		case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT: {
			var = new Variable(fCreateGraph(
				init.float_data().data(), init.float_data_size(), F_FLOAT32,
				(const unsigned long *)init.dims().data(), init.dims().size()));
			break;
		}
		case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE: {
			var = new Variable(fCreateGraph(
				init.double_data().data(), init.double_data_size(), F_FLOAT64,
				(const unsigned long *)init.dims().data(), init.dims().size()));
			break;
		}
		case onnx::TensorProto_DataType::TensorProto_DataType_INT32: {
			var = new Variable(fCreateGraph(
				init.int32_data().data(), init.int32_data_size(), F_INT32,
				(const unsigned long *)init.dims().data(), init.dims().size()));
			break;
		}
		case onnx::TensorProto_DataType::TensorProto_DataType_INT64: {
			var = new Variable(fCreateGraph(
				init.int64_data().data(), init.int64_data_size(), F_INT64,
				(const unsigned long *)init.dims().data(), init.dims().size()));
			break;
		}
		default:
			flogging(F_ERROR, "Unknown type: " + to_string(init.data_type()));
		}
		layers.insert({init.name(), var});
		weights.push_back(var);
	}
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
			} else if (in == "data") {
				auto inn = new InputNode();
				layers.insert({"data", inn});
				x->incoming.push_back(inn);
				inn->outgoing.push_back(x);
			} else
				flogging(F_WARNING, "Unknown input: " + in);
		}
	}
	// retreive input and output names
	GraphModel *res = new GraphModel();
	res->input = vector<InputNode *>(graph.input_size());
	for (int i = 0; i < graph.input_size(); i++) {
		if (!layers.contains(graph.input(i).name()))
			flogging(F_ERROR, "Unknown layer: " + graph.input(i).name());
		if (InputNode *in =
				dynamic_cast<InputNode *>(layers[graph.input(i).name()])) {
			res->input[i] = in;
		} else {
			res->input[i] = new InputNode();
			res->input[i]->outgoing.push_back(layers[graph.input(i).name()]);
		}
	}
	res->output = vector<LayerGraph *>(graph.output_size());
	for (int i = 0; i < graph.output_size(); i++) {
		if (!layers.contains(graph.output(i).name()))
			flogging(F_WARNING, "Unknown layer: " + graph.output(i).name());
		res->output[i] = layers[graph.output(i).name()];
	}
	res->weights = weights;
	return res;
}
std::string GraphModel::serialize_onnx() {
	using namespace onnx;
	ModelProto model;
	GraphProto *graph = model.mutable_graph();
	for (InputNode *in : this->input) {
		auto onnx_in = graph->add_input();
		onnx_in->set_name(in->name);
	}
	int variable = 0;
	for (Variable *w : weights) {
		FGraphNode *node = w->node;
		fExecuteGraph(node);
		FResultData *data = node->result_data;
		TensorProto &proto = *graph->add_initializer();
		proto.set_name(w->name);
		for (int i = 0; i < node->operation.dimensions; i++)
			proto.add_dims(node->operation.shape[i]);
		switch (node->operation.data_type) {
		case F_INT32:
			proto.set_data_type(
				onnx::TensorProto_DataType::TensorProto_DataType_INT32);
			proto.set_raw_data(data->data, sizeof(int) * data->num_entries);
			break;
		case F_INT64:
			proto.set_data_type(
				onnx::TensorProto_DataType::TensorProto_DataType_INT64);
			proto.set_raw_data(data->data, sizeof(long) * data->num_entries);
			break;
		case F_FLOAT32:
			proto.set_data_type(
				onnx::TensorProto_DataType::TensorProto_DataType_FLOAT);
			proto.set_raw_data(data->data, sizeof(float) * data->num_entries);
			break;
		case F_FLOAT64:
			proto.set_data_type(
				onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE);
			proto.set_raw_data(data->data, sizeof(double) * data->num_entries);
			break;
		}
	}
	{
		using namespace std;
		list<LayerGraph *> todo;
		set<LayerGraph *> visited;
		todo.insert(todo.end(), input.begin(), input.end());
		while (!todo.empty()) {
			LayerGraph *curr = todo.front();
			visited.insert(curr);
			todo.pop_front();
			if (!dynamic_cast<InputNode *>(curr) &&
				!dynamic_cast<Variable *>(curr)) {
				onnx::NodeProto *node = graph->add_node();
				curr->deserialize_to_onnx(node);
				node->set_name(curr->name);
				for (LayerGraph *in : curr->incoming)
					node->add_input(in->name);
				node->add_output(curr->name);
			}
			for (LayerGraph *layer : curr->outgoing) {
				if (!visited.contains(layer)) {
					auto ex = std::find(todo.begin(), todo.end(), layer);
					if (ex != todo.end())
						todo.erase(ex);
					todo.push_back(layer);
				}
			}
		}
	}
	graph->set_name("flint_graph");
	for (LayerGraph *out : this->output) {
		auto onnx_out = graph->add_output();
		onnx_out->set_name(out->name);
	}
	return model.SerializeAsString();
}

FGraphNode *GraphModel::operator()(FGraphNode *in) {
	return this->operator()(std::vector{in})[0];
}
std::vector<FGraphNode *> GraphModel::operator()(std::vector<FGraphNode *> in) {
	using namespace std;
	list<LayerGraph *> todo;
	set<LayerGraph *> visited;
	//	set<FGraphNode *> holding = {in};
	for (int i = 0; i < in.size(); i++) {
		InputNode *inp = input[i];
		inp->nodes.clear();
		inp->nodes.push_back(in[i]);
		inp->forward();
		todo.insert(todo.begin(), inp->outgoing.begin(), inp->outgoing.end());
		visited.insert(inp);
	}
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
			curr->forward();
			visited.insert(curr);
			if (std::find(output.begin(), output.end(), curr) != output.end()) {
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
	}
	vector<FGraphNode *> outputs;
	for (LayerGraph *outg : output) {
		outputs.insert(outputs.begin(), outg->output.begin(),
					   outg->output.end());
		for (FGraphNode *out : outg->output)
			out->reference_counter--;
	}
	return outputs;
}
GraphModel *GraphModel::sequential(std::vector<LayerGraph *> list) {
	GraphModel *model = new GraphModel();
	model->input = {new InputNode()};
	LayerGraph *prev = model->input[0];
	for (LayerGraph *node : list) {
		prev->outgoing.push_back(node);
		for (LayerGraph *pw : prev->outgoing) {
			if (Variable *w = dynamic_cast<Variable *>(pw))
				model->weights.push_back(w);
		}
		node->incoming.insert(node->incoming.begin(), prev);
	}
	model->output = {prev};
	return model;
}
GraphModel *GraphModel::from_output(LayerGraph *output) {
	GraphModel *model = new GraphModel();
	model->output = {output};
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
			model->input.push_back(i);
		}
		for (LayerGraph *n : curr->incoming) {
			if (visited.insert(n).second) {
				todo.push_back(n);
			}
		}
	}
	return model;
}
