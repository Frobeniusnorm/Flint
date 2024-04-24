#ifndef ONNX_LAYERS
#define ONNX_LAYERS
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

	private:
		bool marked_for_deletion = false; // for destructor
};
struct Variable : public LayerGraph {
		FGraphNode *node = nullptr;
		Variable() : LayerGraph(1) {}
		Variable(FGraphNode *node) : LayerGraph(1), node(node) {
			node->reference_counter++;
		}
		~Variable() {
			if (node) {
				node->reference_counter--;
			}
		}
		void forward() override { output[0] = node; }
};
struct InputNode : public LayerGraph {
		std::vector<FGraphNode *> nodes;
		bool transpose_channel = false;
		InputNode() : LayerGraph(1) {}
		InputNode(FGraphNode *node) : LayerGraph(1), nodes{node} {}
		InputNode(std::vector<FGraphNode *> nodes)
			: LayerGraph(1), nodes(nodes) {}
		void forward() override {
			output.clear();
			for (FGraphNode *node : nodes)
				output.push_back(node);
		}
};
struct Relu : public LayerGraph {
		Relu() : LayerGraph(1) {}
		Relu(LayerGraph *in) : LayerGraph(in) {}
		void forward() override;
};
struct Flatten : public LayerGraph {
		Flatten() : LayerGraph(1) {}
		Flatten(LayerGraph *in) : LayerGraph(in) {}
		void forward() override;
};
struct Add : public LayerGraph {
		Add() : LayerGraph(1) {}
		Add(LayerGraph *a, LayerGraph* b) : LayerGraph({a, b}) {}
		void forward() override;
};
struct Convolve : public LayerGraph {
		std::vector<unsigned int> stride, padding;
		Convolve() : LayerGraph(1) {}
		Convolve(std::vector<unsigned int> stride,
				 std::vector<unsigned int> padding)
			: LayerGraph(1), stride(stride), padding(padding) {}
		Convolve(std::vector<unsigned int> stride,
				 std::vector<unsigned int> padding, LayerGraph *in,
				 Variable *kernel, Variable *bias)
			: LayerGraph({in, kernel, bias}), stride(stride), padding(padding) {
		}
		void forward() override;
};
struct MaxPool : public LayerGraph {
		std::vector<size_t> kernel_shape;
		std::vector<unsigned int> stride, padding;
		MaxPool() : LayerGraph(1) {}
		MaxPool(std::vector<size_t> kernel_shape,
				std::vector<unsigned int> stride,
				std::vector<unsigned int> padding)
			: LayerGraph(1), kernel_shape(kernel_shape), stride(stride),
			  padding(padding) {}
		MaxPool(std::vector<size_t> kernel_shape,
				std::vector<unsigned int> stride,
				std::vector<unsigned int> padding, LayerGraph *in)
			: LayerGraph(in), kernel_shape(kernel_shape), stride(stride),
			  padding(padding) {}
		void forward() override;
};
struct AvgPool : public LayerGraph {
		std::vector<size_t> kernel_shape;
		std::vector<unsigned int> stride, padding;
		AvgPool() : LayerGraph(1) {}
		AvgPool(std::vector<size_t> kernel_shape,
				std::vector<unsigned int> stride,
				std::vector<unsigned int> padding)
			: LayerGraph(1), kernel_shape(kernel_shape), stride(stride),
			  padding(padding) {}
		AvgPool(std::vector<size_t> kernel_shape,
				std::vector<unsigned int> stride,
				std::vector<unsigned int> padding, LayerGraph *in)
			: LayerGraph(in), kernel_shape(kernel_shape), stride(stride),
			  padding(padding) {}
		void forward() override;
};
struct GlobalAvgPool : public LayerGraph {
		GlobalAvgPool() : LayerGraph(1) {}
		GlobalAvgPool(LayerGraph *in) : LayerGraph(in) {}
		void forward() override;
};
struct BatchNorm : public LayerGraph {
		BatchNorm(float alpha = 0.8) : LayerGraph(1), alpha(alpha) {}
		BatchNorm(LayerGraph *in, Variable *gamma, Variable *beta,
				  float alpha = 0.8)
			: LayerGraph({in, gamma, beta}), alpha(alpha) {}
		void forward() override;
		float alpha;
};
struct Connected : public LayerGraph {
		Connected() : LayerGraph(1) {}
		Connected(LayerGraph *in, Variable *weight, Variable *bias = nullptr,
				  bool transposeA = false, bool transposeB = false)
			: LayerGraph({in, weight, bias}), transposeA(transposeA),
			  transposeB(transposeB) {}
		void forward() override;
		bool transposeA = false;
		bool transposeB = false;
};

#endif
