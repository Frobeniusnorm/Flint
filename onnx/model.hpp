#ifndef ONNX_MODEL
#define ONNX_MODEL
#include "../flint.h"
#include "layers/layers.hpp"
#include <list>
struct GraphModel {
		InputNode *input;
		LayerGraph *output;
    GraphModel() = default;
		FGraphNode *operator()(FGraphNode *in) {
			input->output.clear();
			input->output.push_back(in);
			using namespace std;
			list<LayerGraph *> todo;
			list<FGraphNode *> holding = {in};
			in->reference_counter++;
			todo.insert(todo.begin(), input->outgoing.begin(), input->outgoing.end());
			todo.push_back(nullptr); // as a marker that the holding reference
									 // can be removed
			while (!todo.empty()) {
				LayerGraph *curr = todo.front();
				if (curr) {
					todo.pop_front();
					curr->forward();
					// ensure that the output is not consumed
					for (FGraphNode *out : curr->output) {
						holding.push_back(out);
						out->reference_counter++;
					}
					// add the children bfs
					for (LayerGraph *layer : curr->outgoing)
						todo.push_back(layer);
				} else {
					for (FGraphNode *h : holding) {
						h->reference_counter--;
						// maybe it is already no longer needed
						fFreeGraph(h);
					}
					holding.clear();
					todo.push_back(nullptr);
				}
			}
			return output->output[0];
		}
    static GraphModel load_model(std::string path);
		// TODO training
};

#endif
