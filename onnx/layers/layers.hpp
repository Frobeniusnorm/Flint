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

		virtual std::vector<FGraphNode *> collect_weights() { return {}; }
		virtual void set_weights(std::vector<FGraphNode *> weights) {}
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

#endif
