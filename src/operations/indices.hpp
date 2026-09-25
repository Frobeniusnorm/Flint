/* Copyright 2026 David Schwarzbeck
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
#pragma once
// This files manages the datastructures for the index dataflow graph for the
// OpenCL implementation
#include "flint/flint.h"
#include <cstddef>
#include <string>
#include <vector>
struct IndexCoeeficient {
		FType c_type;
};
/* Some OCL Variable */
struct IndexAtom {
		int id; // reference from OclCompilerState
		std::pair<size_t, size_t>
			bound;		  // size bound (for modulo or known bound constants)
		std::string name; // a variable needs a name
};
struct IndexMap;
/**
 * Holds an expression a + c1 * atom1 + c2 * atom2 + ...
 */
struct IndexExpr {
		long a;
		std::vector<std::pair<long, int>> exprs; // (c_i, atom_i)
		std::pair<size_t, size_t> bound; // size bound (for modulo or known
										 // bound constants), derived from atoms
		bool operator==(const IndexExpr other) const {
			return a == other.a && exprs == other.exprs && bound == other.bound;
		}
};
/**
 * The resulting index for a node.
 */
struct IndexMap {
		std::vector<IndexExpr>
			expr_per_dim; // index expression per dimension. Outermost dimension
						  // is the first entry (just like for an FOperation's
						  // shape).
		/**
		 * Flattens this multidimensional index map to a flat index expression.
		 */
		IndexExpr flatten(const size_t *shape, int dims) {
			if (dims != expr_per_dim.size()) {
				flogging(F_ERROR, "Index flatten: Operation has different "
								  "number of dimensions than the IndexMap.");
			}
			IndexExpr result = expr_per_dim[dims - 1];
			size_t coeff = shape[dims - 1];
			for (int i = dims - 2; i >= 0; i--) {
				IndexExpr curr = expr_per_dim[i];
				result.a += curr.a * coeff;
				result.exprs.reserve(result.exprs.size() + curr.exprs.size());
				for (auto exp : curr.exprs) {
					exp.first *= coeff;
					result.exprs.push_back(exp);
				}
				coeff *= shape[i];
			}
			return result;
		}
		/**
		 * Reconstructs the multidimensional index map from a flat one.
		 */
		static IndexMap split(IndexExpr expr, const size_t *shape, int dims) {
			IndexMap result;
			result.expr_per_dim.resize(dims);
			size_t coeff = 1;
			for (int i = dims - 1; i >= 0; i--) {
				size_t n_coeff = coeff * shape[i];
				IndexExpr curr;
				curr.a = (expr.a % n_coeff) / coeff;
				curr.exprs.reserve(expr.exprs.size());
				for (auto e : expr.exprs) {
					e.first = (e.first % n_coeff) / coeff;
					if (e.first != 0)
						curr.exprs.push_back(e);
				}
				coeff = n_coeff;
				result.expr_per_dim[i] = curr;
			}

			return result;
		}
		bool operator==(const IndexMap other) const {
			return expr_per_dim == other.expr_per_dim;
		}
};
