/* Copyright 2023 David Schwarzbeck
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */
#include "../layers.hpp"

struct Flatten : public UntrainableLayer {
		static constexpr unsigned int transform_dimensionality(unsigned int n) {
			return 2;
		}
		/** Flattens every feature axis into one single axis, does not touch the
		 * batch-axis (the first) */
		template <typename T, unsigned int n>
		Tensor<T, 2> forward(Tensor<T, n> &in) {
			if constexpr (n == 2)
				return in;
			else {
        FGraphNode* node = in.get_graph_node();
        // consume the node (i.e. remove handle from wrapper)
        in.set_graph_node(nullptr);
        node->reference_counter--;
        while (node->operation.dimensions > 2) {
          node = fflatten_dimension(node, node->operation.dimensions - 1);
        }
				return Tensor<T, 2>(node);
      }
		}
		std::string name() override { return "Flatten"; }
};
