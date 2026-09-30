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
#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
struct IndexCoeeficient {
		FType c_type;
};
/* Some OCL Variable */
struct IndexAtom {
		int id; // reference from OclCompilerState
		std::pair<long, long>
			bound;		  // smallest and largest value it can take, inclusive
		std::string name; // a variable needs a name
};
/**
 * Holds an expression a + c1 * atom1 + c2 * atom2 + ...
 */
struct IndexExpr {
		long a;
		std::vector<std::pair<long, int>> exprs; // (c_i, atom_i)
		std::pair<long, long> bound;			 // derived, see `derive_bound`
		bool operator==(const IndexExpr other) const {
			return a == other.a && exprs == other.exprs;
		}
		/**
		 * Emits the expression, e.g. `(3 + 2 * index_dim0 - i5)`. It is
		 * parenthesized if it has more than one summand, so that it can be
		 * used inside a larger expression.
		 */
		std::string to_code(const std::vector<IndexAtom> &atoms) const {
			if (exprs.empty())
				return std::to_string(a);
			std::string res = a != 0 ? std::to_string(a) : "";
			for (const auto &[c, atom] : exprs) {
				const long factor = c < 0 ? -c : c;
				if (!res.empty())
					res += c < 0 ? " - " : " + ";
				else if (c < 0)
					res += "-";
				res += factor == 1
						   ? atoms[atom].name
						   : std::to_string(factor) + " * " + atoms[atom].name;
			}
			return exprs.size() == 1 && a == 0 ? res : "(" + res + ")";
		}
		void derive_bound(const std::vector<IndexAtom> &atoms) {
			bound = {a, a};
			for (const auto &[c, atom] : exprs) {
				const std::pair<long, long> &ab = atoms[atom].bound;
				bound.first += c >= 0 ? c * ab.first : c * ab.second;
				bound.second += c >= 0 ? c * ab.second : c * ab.first;
			}
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
		IndexExpr flatten(const size_t *shape, int dims,
						  const std::vector<IndexAtom> &atoms) {
			if (dims != expr_per_dim.size()) {
				flogging(F_ERROR, "Index flatten: Operation has different "
								  "number of dimensions than the IndexMap.");
			}
			IndexExpr result = expr_per_dim[dims - 1];
			long coeff = shape[dims - 1];
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
			result.derive_bound(atoms);
			return result;
		}
		/**
		 * Reconstructs the multidimensional index map from a flat one.
		 */
		static IndexMap split(IndexExpr expr, const size_t *shape, int dims,
							  const std::vector<IndexAtom> &atoms) {
			IndexMap result;
			result.expr_per_dim.resize(dims);
			long coeff = 1;
			for (int i = dims - 1; i >= 0; i--) {
				const long n_coeff = coeff * (long)shape[i];
				IndexExpr curr;
				curr.a = (expr.a % n_coeff) / coeff;
				curr.exprs.reserve(expr.exprs.size());
				for (auto e : expr.exprs) {
					e.first = (e.first % n_coeff) / coeff;
					if (e.first != 0)
						curr.exprs.push_back(e);
				}
				curr.derive_bound(atoms);
				curr.bound.first = std::max(curr.bound.first, 0l);
				curr.bound.second =
					std::min(curr.bound.second, (long)shape[i] - 1);
				coeff = n_coeff;
				result.expr_per_dim[i] = curr;
			}

			return result;
		}
		bool operator==(const IndexMap other) const {
			return expr_per_dim == other.expr_per_dim;
		}
};
/**
 * The coordinates a parameter is read at when it is read at the same position
 * as the node itself. Its shape has to be the trailing part of the node's
 * shape, then the leading coordinates are dropped and a dimension the parameter
 * has as size 1 becomes the coordinate 0, which is how a smaller parameter is
 * broadcast over the node. Returns false if the shapes do not fit that way,
 * e.g. for the inverse broadcasting modes.
 */
inline bool trailing_map(const IndexMap &node_map, const size_t *node_shape,
						 int node_dims, const size_t *pred_shape, int pred_dims,
						 IndexMap &result) {
	if (pred_dims > node_dims)
		return false;
	const int offset = node_dims - pred_dims;
	result.expr_per_dim.resize(pred_dims);
	for (int i = 0; i < pred_dims; i++) {
		if (pred_shape[i] == 1)
			result.expr_per_dim[i] = IndexExpr{0, {}, {0, 0}};
		else if (pred_shape[i] == node_shape[i + offset])
			result.expr_per_dim[i] = node_map.expr_per_dim[i + offset];
		else
			return false;
	}
	return true;
}
