#ifndef ONNX_LAYERS
#define ONNX_LAYERS
#include "onnx.proto3.pb.h"
#include <string>
#define FLINT_DEBUG
#include "flint.h"
#include <vector>
// TODO adapt the data s.t. channels are switched to back to remove the
// transpositions
struct LayerGraph {
		std::vector<LayerGraph *> incoming; // incoming edges in graph
		std::vector<LayerGraph *> outgoing; // incoming edges in graph
		std::vector<FGraphNode *> output;	// result of forward
		bool training = false;
		std::string name;
		LayerGraph() = default;
		virtual ~LayerGraph() {
			marked_for_deletion = true;
			for (LayerGraph *lg : incoming)
				if (lg && !lg->marked_for_deletion) {
					lg->marked_for_deletion = true;
					delete lg;
				}
			for (LayerGraph *lg : outgoing)
				if (lg && !lg->marked_for_deletion) {
					lg->marked_for_deletion = true;
					delete lg;
				}
		}
		LayerGraph(size_t reserved_output_slots)
			: output(reserved_output_slots) {
			for (size_t i = 0; i < reserved_output_slots; i++)
				output[i] = nullptr;
		}
		LayerGraph(LayerGraph *in, size_t reserved = 1) : LayerGraph(reserved) {
			incoming.push_back(in);
			in->outgoing.push_back(this);
		}
		LayerGraph(std::vector<LayerGraph *> in, size_t reserved = 1)
			: LayerGraph(reserved) {
			for (LayerGraph *i : in) {
				if (i) {
					incoming.push_back(i);
					i->outgoing.push_back(this);
				}
			}
		}
		/**
		 * Computes the layer output in `output` by the output of the
		 * previous layers. The framework makes sure that the output of
		 * the incoming layers exists.
		 */
		virtual void forward() = 0;
		virtual void deserialize_to_onnx(onnx::NodeProto *) {
			flogging(F_WARNING,
					 "Node is deserialized to onnx without an implementation.");
		};
		virtual std::vector<std::vector<size_t>>
		propagate_shape(const std::vector<std::vector<size_t>> &input) {
			return input;
		}

	private:
		bool marked_for_deletion = false; // for destructor
};
struct Variable : public LayerGraph {
		static int variable_no;
		FGraphNode *node = nullptr;
		Variable() : LayerGraph(1) {
			name = "Variable" + std::to_string(variable_no++);
		}
		Variable(FGraphNode *node) : LayerGraph(1), node(node) {
			node->reference_counter++;
			name = "Variable" + std::to_string(variable_no++);
		}
		~Variable() {
			if (node) {
				node->reference_counter--;
			}
		}
		void forward() override { output[0] = node; }
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			// already included as an initializer
		}
		std::vector<std::vector<size_t>> propagate_shape(
			const std::vector<std::vector<size_t>> &input) override {
			return {std::vector<size_t>(node->operation.shape,
										node->operation.shape +
											node->operation.dimensions)};
		}
};
struct InputNode : public LayerGraph {
		static int data_no;
		std::vector<FGraphNode *> nodes;
		bool transpose_channel = false;
		InputNode() : LayerGraph(1) {
			int no = data_no++;
			name = "data";
			if (no)
				name += std::to_string(no);
		}
		InputNode(int no_nodes) : LayerGraph(no_nodes) {
			int no = data_no++;
			name = "data";
			if (no)
				name += std::to_string(no);
		}
		void forward() override {
			output.clear();
			for (FGraphNode *node : nodes)
				output.push_back(node);
		}
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			// only referenced by its name
		}
};
struct Relu : public LayerGraph {
		static int relu_no;
		Relu() : LayerGraph(1) {}
		Relu(LayerGraph *in) : LayerGraph(in) {
			name = "Relu" + std::to_string(relu_no++);
		}
		void forward() override;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("Relu");
		}
};
struct Flatten : public LayerGraph {
		static int flatten_no;
		Flatten() : LayerGraph(1) {
			name = "Flatten" + std::to_string(flatten_no++);
		}
		Flatten(LayerGraph *in) : LayerGraph(in) {}
		void forward() override;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("Flatten");
		}
};
struct Add : public LayerGraph {
		static int add_no;
		Add() : LayerGraph(1) { name = "Add" + std::to_string(add_no++); }
		Add(LayerGraph *a, LayerGraph *b) : LayerGraph({a, b}) {}
		void forward() override;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("Add");
		}
};
static void deserialize_stride_and_padding(onnx::NodeProto *node,
										   std::vector<unsigned int> &stride,
										   std::vector<unsigned int> &padding) {
	auto attr = node->add_attribute();
	attr->set_name("pads");
	for (unsigned int p : padding)
		attr->add_ints(p);
	attr = node->add_attribute();
	attr->set_name("strides");
	for (unsigned int s : stride)
		attr->add_ints(s);
}
static std::vector<size_t>
convolve_shape_transform(const std::vector<size_t> in,
						 const std::vector<size_t> filter,
						 const std::vector<unsigned int> &padding,
						 const std::vector<unsigned int> &stride) {
	using namespace std;
	vector<size_t> out_shape(in.size());
	out_shape[0] = in[0];						 // step size
	out_shape[out_shape.size() - 1] = filter[0]; // filters
	for (int i = 1; i < in.size() - 1; i++) {
		size_t padded_shape = in[i];
		if (padding.size() != 0 && i > 0 && i < in.size() - 1) {
			padded_shape += padding[i - 1] + padding[i - 1 + in.size() - 2];
		}
		size_t window_size = padded_shape - filter[i] + 1;
		window_size = window_size % filter[i] == 0
						  ? window_size / stride[i - 1]
						  : window_size / stride[i - 1] + 1;
		out_shape[i] = window_size;
	}
	return out_shape;
}
struct Convolve : public LayerGraph {
		static int conv_no;
		std::vector<unsigned int> stride, padding;
		Convolve() : LayerGraph(1) {
			name = "Conv" + std::to_string(conv_no++);
		}
		Convolve(std::vector<unsigned int> stride,
				 std::vector<unsigned int> padding)
			: LayerGraph(1), stride(stride), padding(padding) {
			name = "Conv" + std::to_string(conv_no++);
		}
		Convolve(std::vector<unsigned int> stride,
				 std::vector<unsigned int> padding, LayerGraph *in,
				 Variable *kernel, Variable *bias)
			: LayerGraph({in, kernel, bias}), stride(stride), padding(padding) {
			name = "Conv" + std::to_string(conv_no++);
		}
		void forward() override;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("Conv");
			deserialize_stride_and_padding(node, stride, padding);
		}
		std::vector<std::vector<size_t>> propagate_shape(
			const std::vector<std::vector<size_t>> &input) override {
			using namespace std;
			return {
				convolve_shape_transform(input[0], input[1], padding, stride)};
		}
};
static void deserialize_kernel_shape(onnx::NodeProto *node,
									 std::vector<size_t> &kernel) {
	auto attr = node->add_attribute();
	attr->set_name("kernel_shape");
	for (size_t s : kernel)
		attr->add_ints(s);
}
struct MaxPool : public LayerGraph {
		static int mpool_no;
		std::vector<size_t> kernel_shape;
		std::vector<unsigned int> stride, padding;
		MaxPool() : LayerGraph(1) {
			name = "MaxPool" + std::to_string(mpool_no++);
		}
		MaxPool(std::vector<size_t> kernel_shape,
				std::vector<unsigned int> stride,
				std::vector<unsigned int> padding)
			: LayerGraph(1), kernel_shape(kernel_shape), stride(stride),
			  padding(padding) {
			name = "MaxPool" + std::to_string(mpool_no++);
		}
		MaxPool(std::vector<size_t> kernel_shape,
				std::vector<unsigned int> stride,
				std::vector<unsigned int> padding, LayerGraph *in)
			: LayerGraph(in), kernel_shape(kernel_shape), stride(stride),
			  padding(padding) {
			name = "MaxPool" + std::to_string(mpool_no++);
		}
		void forward() override;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("MaxPool");
			deserialize_stride_and_padding(node, stride, padding);
			deserialize_kernel_shape(node, kernel_shape);
		}
		std::vector<std::vector<size_t>> propagate_shape(
			const std::vector<std::vector<size_t>> &input) override {
			using namespace std;
			return {convolve_shape_transform(input[0], kernel_shape, padding,
											 stride)};
		}
};
struct AvgPool : public LayerGraph {
		static int apool_no;
		std::vector<size_t> kernel_shape;
		std::vector<unsigned int> stride, padding;
		AvgPool() : LayerGraph(1) {
			name = "AvgPool" + std::to_string(apool_no++);
		}
		AvgPool(std::vector<size_t> kernel_shape,
				std::vector<unsigned int> stride,
				std::vector<unsigned int> padding)
			: LayerGraph(1), kernel_shape(kernel_shape), stride(stride),
			  padding(padding) {
			name = "AvgPool" + std::to_string(apool_no++);
		}
		AvgPool(std::vector<size_t> kernel_shape,
				std::vector<unsigned int> stride,
				std::vector<unsigned int> padding, LayerGraph *in)
			: LayerGraph(in), kernel_shape(kernel_shape), stride(stride),
			  padding(padding) {
			name = "AvgPool" + std::to_string(apool_no++);
		}
		void forward() override;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("AvgPool");
			deserialize_stride_and_padding(node, stride, padding);
			deserialize_kernel_shape(node, kernel_shape);
		}
		std::vector<std::vector<size_t>> propagate_shape(
			const std::vector<std::vector<size_t>> &input) override {
			using namespace std;
			return {convolve_shape_transform(input[0], kernel_shape, padding,
											 stride)};
		}
};
struct GlobalAvgPool : public LayerGraph {
		static int gapool_no;
		GlobalAvgPool() : LayerGraph(1) {
			name = "GlobalAvgPool" + std::to_string(gapool_no++);
		}
		GlobalAvgPool(LayerGraph *in) : LayerGraph(in) {
			name = "GlobalAvgPool" + std::to_string(gapool_no++);
		}
		void forward() override;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("GlobalAveragePool");
		}
		std::vector<std::vector<size_t>> propagate_shape(
			const std::vector<std::vector<size_t>> &input) override {
			using namespace std;
			const int dims = input[0].size();
			vector<size_t> rank_shape(dims, 1);
			rank_shape[0] = input[0][0];
			rank_shape[1] = input[0][1];
			return {rank_shape};
		}
};
struct BatchNorm : public LayerGraph {
		// TODO dynamically determine if running mean and variance should be
		// used as additional outputs
		static int bnorm_no;
		BatchNorm(float alpha = 0.8) : LayerGraph(1), alpha(alpha) {
			name = "BatchNorm" + std::to_string(bnorm_no++);
		}
		BatchNorm(LayerGraph *in, Variable *gamma, Variable *beta,
				  float alpha = 0.8)
			: LayerGraph({in, gamma, beta}), alpha(alpha) {
			name = "BatchNorm" + std::to_string(bnorm_no++);
		}
		void forward() override;
		float alpha;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("BatchNormalization");
			auto attr_alpha = node->add_attribute();
			attr_alpha->set_name("momentum");
			attr_alpha->set_f(alpha);
		}
};
struct Connected : public LayerGraph {
		static int connected_no;
		Connected() : LayerGraph(1) {
			name = "Connected" + std::to_string(connected_no++);
		}
		Connected(LayerGraph *in, Variable *weight, Variable *bias = nullptr,
				  bool transposeA = false, bool transposeB = false)
			: LayerGraph({in, weight, bias}), transposeA(transposeA),
			  transposeB(transposeB) {
			name = "Connected" + std::to_string(connected_no++);
		}
		void forward() override;
		bool transposeA = false;
		bool transposeB = false;
		void deserialize_to_onnx(onnx::NodeProto *node) override {
			node->set_op_type("Gemm");
			auto attr_a = node->add_attribute();
			attr_a->set_name("transA");
			attr_a->set_i(transposeA ? 1 : 0);
			auto attr_b = node->add_attribute();
			attr_b->set_name("transB");
			attr_b->set_i(transposeB ? 1 : 0);
		}
		std::vector<std::vector<size_t>> propagate_shape(
			const std::vector<std::vector<size_t>> &input) override {
			using namespace std;
			const vector<size_t> &img = input[0];
			const vector<size_t> &filter = input[1];
			vector<size_t> output(img);
			size_t n = transposeA ? output[output.size() - 1]
								  : output[output.size() - 2];
			size_t k = transposeB ? filter[filter.size() - 2]
								  : filter[filter.size() - 1];
			output[output.size() - 2] = n;
			output[output.size() - 1] = k;
			return {output};
		}
};

#endif
