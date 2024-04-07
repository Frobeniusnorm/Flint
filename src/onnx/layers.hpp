#ifndef ONNX_LAYERS
#define ONNX_LAYERS
#define FLINT_DEBUG
#include "flint.h"
#include <vector>
struct LayerGraph {
		std::vector<LayerGraph *> incoming; // incoming edges in graph
		std::vector<LayerGraph *> outgoing; // incoming edges in graph
		std::vector<FGraphNode *> output;	// result of forward
		bool training =
			false; // wheather or not the model is in training or in testing

		LayerGraph() = default;
		LayerGraph(size_t reserved_output_slots)
			: output(reserved_output_slots) {
			for (size_t i = 0; i < reserved_output_slots; i++)
				output[i] = nullptr;
		}
		/**
		 * Computes the layer output in `output` by the output of the previous
		 * layers. The framework makes sure that the ouput of the incoming
		 * layers exists.
		 */
		virtual void forward() = 0;
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
		void forward() override;
};
struct Flatten : public LayerGraph {
		Flatten() : LayerGraph(1) {}
		void forward() override;
};
struct Add : public LayerGraph {
		Add() : LayerGraph(1) {}
		void forward() override;
};
struct Convolve : public LayerGraph {
		std::vector<unsigned int> stride, padding;
		Convolve() : LayerGraph(1) {}
		Convolve(std::vector<unsigned int> stride,
				 std::vector<unsigned int> padding)
			: LayerGraph(1), stride(stride), padding(padding) {}
		void forward() override;
};
struct MaxPool : public LayerGraph {
		std::vector<size_t> kernel_shape;
		std::vector<unsigned int> stride, padding;
		MaxPool() : LayerGraph(1) {}
		MaxPool(std::vector<size_t> kernel_shape,
				std::vector<unsigned int> stride,
				std::vector<unsigned int> padding)
			: kernel_shape(kernel_shape), stride(stride), padding(padding) {}
		void forward() override;
};
struct AvgPool : public LayerGraph {
		std::vector<size_t> kernel_shape;
		std::vector<unsigned int> stride, padding;
		AvgPool() : LayerGraph(1) {}
		AvgPool(std::vector<size_t> kernel_shape,
				std::vector<unsigned int> stride,
				std::vector<unsigned int> padding)
			: kernel_shape(kernel_shape), stride(stride), padding(padding) {}
		void forward() override;
};
struct GlobalAvgPool : public LayerGraph {
		GlobalAvgPool() : LayerGraph(1) {}
		void forward() override;
};
struct BatchNorm : public LayerGraph {
  // TODO running mean and variance as parameters !
		BatchNorm(float alpha = 0.8) : LayerGraph(1), alpha(alpha) {}
		void forward() override;
		float alpha;
};
struct Connected : public LayerGraph {
		Connected() : LayerGraph(1) {}
		void forward() override;
};

#endif
