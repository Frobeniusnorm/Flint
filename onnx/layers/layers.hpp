#ifndef ONNX_LAYERS
#define ONNX_LAYERS

#include "flint.h"
#include <vector>
struct LayerGraph {
		std::vector<LayerGraph *> incoming; // incoming edges in graph
		std::vector<FGraphNode *> output;	// result of forward

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
		Convolve(
				 std::vector<unsigned int> stride,
				 std::vector<unsigned int> padding)
			: stride(stride), padding(padding) {}
		void forward() override;
};

#endif
