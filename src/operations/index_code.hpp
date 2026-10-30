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
// Turns the index maps of the OpenCL code generation into code: it owns the
// atoms, decides which part of an address becomes a variable and where that
// variable is declared, and decomposes offsets into coordinates.
#include "../../flint.h"
#include "src/operations/indices.hpp"
#include <cassert>
#include <climits>
#include <list>
#include <map>
#include <string>
#include <vector>
/** Symbolic for `(source / divisor) % modulus` */
struct SplitCoord {
		IndexExpr source;
		long divisor;
		long modulus;
};
/** symbolic Index decleration*/
struct IndexDecl {
		/** depth of the scope it is declared in */
		unsigned int depth;
		/** the operation that registered it */
		unsigned int owner;
		std::string expr;
		std::string var;
		std::string type;
		int atom = -1;
};
struct IndexCode {
		IndexMap index_map;
		/** the coordinates each parameter of that node is read at
		 * see (`OperationImplementation::mutate_index`) */
		std::vector<IndexMap> pred_index_maps;
		/** pool of all index atoms (e.g. index variables or other unknowns) */
		std::vector<IndexAtom> index_atoms;
		/** Declarations of index expressions that are still to be emitted.
		 * Emitted once the depth they live in is opened. */
		std::list<IndexDecl> index_decls;
		/** Number of index declarations (for naming) */
		unsigned int num_index_decls = 0;
		/** Number of coordinate variables (for naming) */
		unsigned int num_coords = 0;
		/** Number of operations that were generated (differentiate operation
		 * from indirect successors)*/
		unsigned int generated_ops = 0;
		/** Coordinates `split_atoms` masked out of one expression, by the id
		 * of the atom that holds them. Striding them back together is the
		 * expression again. */
		std::map<int, SplitCoord> split_coords;
		/**  the smallest type for a variable with the `bounds` */
		static std::string type_of(const std::pair<long, long> &bound) {
			if (bound.first >= 0 && bound.second > INT_MAX)
				return bound.second <= UINT_MAX ? "unsigned int"
												: "unsigned long";
			return bound.first >= INT_MIN && bound.second <= INT_MAX ? "int"
																	 : "long";
		}
		/**
		 * Registers an unknown for the index calculations and returns its id
		 */
		int add_atom(const std::string &name, long lo, long hi, int depth,
					 bool inlined = false) {
			index_atoms.emplace_back(std::pair<long, long>{lo, hi}, name, depth,
									 inlined);
			return index_atoms.size() - 1;
		}
		/** The innermost loop an expression depends on */
		unsigned int depth_of(const IndexExpr &expr) const {
			unsigned int depth = 0;
			for (const auto &[c, atom] : expr.exprs)
				depth = std::max(depth, index_atoms[atom].depth);
			return depth;
		}
		/**
		 * `map` flattened along `shape`.
		 */
		IndexExpr flatten(IndexMap &map, const size_t *shape, int dims) {
			long stride = 1;
			int first = dims; // first dimension of the block
			const IndexExpr *source = nullptr;
			for (int i = dims - 1; i >= 0; i--) {
				const IndexExpr &coord = map.expr_per_dim[i];
				if (coord.exprs.empty() && coord.a == 0) {
					stride *= (long)shape[i];
					first = i;
					continue;
				}
				if (coord.a != 0 || coord.exprs.size() != 1 ||
					coord.exprs[0].first != 1)
					break;
				const auto split = split_coords.find(coord.exprs[0].second);
				if (split == split_coords.end() ||
					split->second.divisor != stride ||
					split->second.modulus != (long)shape[i] ||
					(source && !(*source == split->second.source)))
					break;
				source = &split->second.source;
				stride *= (long)shape[i];
				first = i;
			}
			// only if the block holds all of the expression, a part of it
			// would need the modulo back
			if (!source || source->bound.first < 0 ||
				source->bound.second >= stride)
				return map.flatten(shape, dims);
			IndexExpr result = *source;
			for (int i = first - 1; i >= 0; i--) {
				const IndexExpr &coord = map.expr_per_dim[i];
				result.a += coord.a * stride;
				for (auto exp : coord.exprs) {
					exp.first *= stride;
					result.exprs.push_back(exp);
				}
				result.bound.first += coord.bound.first * stride;
				result.bound.second += coord.bound.second * stride;
				stride *= (long)shape[i];
			}
			return result;
		}
		/** Emits `expr` as code */
		std::string to_code(const IndexExpr &expr) {
			if (expr.exprs.empty())
				return std::to_string(expr.a);
			// the atoms of one depth belong into one part
			std::map<unsigned int, IndexExpr> per_depth;
			for (const auto &[c, atom] : expr.exprs) {
				const IndexAtom &ia = index_atoms[atom];
				assert(ia.depth <= index_map.current_depth);
				per_depth[ia.depth].exprs.push_back({c, atom});
			}
			// the offset is constant in every loop, so it is calculated with
			// the part that leaves the most of them
			per_depth.begin()->second.a = expr.a;
			std::string res;
			for (auto &[depth, part] : per_depth) {
				part.derive_bound(index_atoms);
				if (!res.empty())
					res += " + ";
				// a part that reads a value of its own scope may not be
				// calculated in front of it
				bool inlined = false;
				for (const auto &[c, atom] : part.exprs)
					inlined |= index_atoms[atom].inlined;
				res += inlined ? part.to_code(index_atoms)
							   : declare_index(depth, part);
			}
			// the result is used inside of larger expressions
			return per_depth.size() > 1 ? "(" + res + ")" : res;
		}
		/**
		 * Adds a variable declaration for `part` and makes sure it is emitted
		 * on `depth`. Returns the name of the variable.
		 */
		std::string declare_index(unsigned int depth, const IndexExpr &part) {
			// a part that is a single atom is a variable already
			if (part.a == 0 && part.exprs.size() == 1 &&
				part.exprs[0].first == 1)
				return index_atoms[part.exprs[0].second].name;
			const std::string expr = part.to_code(index_atoms);
			// reuse expressions
			for (const IndexDecl &decl : index_decls)
				if (decl.expr == expr)
					return decl.var;
			// else declare new one
			const std::string var = "i" + std::to_string(depth) + "_" +
									std::to_string(num_index_decls++);
			index_decls.push_back(
				{depth, generated_ops, expr, var, type_of(part.bound)});
			return var;
		}
		/**
		 * Declares `expr` as a variable of `type` in the scope of `depth` and
		 * registers it as an atom of these bounds. Returns that atom.
		 */
		int declare_coord(unsigned int depth, const std::string &type,
						  const std::string &expr, long lo, long hi) {
			for (const IndexDecl &decl : index_decls)
				if (decl.expr == expr && decl.atom >= 0)
					return decl.atom;
			const std::string var = "c" + std::to_string(num_coords++);
			const int atom = add_atom(var, lo, hi, depth);
			index_decls.push_back(
				{depth, generated_ops, expr, var, type, atom});
			return atom;
		}
		/** Declares the variable of an existing `atom` as `expr`, in the scope
		 * of the atom's depth */
		void declare_atom(int atom, const std::string &expr) {
			const IndexAtom &ia = index_atoms[atom];
			index_decls.push_back({ia.depth, generated_ops, expr, ia.name,
								   type_of(ia.bound), atom});
		}
		static std::string declaration(const IndexDecl &decl) {
			return decl.type + " " + decl.var + " = " + decl.expr + ";\n";
		}
};
/**
 * Decomposes a flat element offset into one coordinate per dimension of
 * `shape`. Where that can not be shown the coordinate is calculated into a
 * variable, unless `free_only` is set, which gives an empty map instead and
 * emits nothing. A term whose coefficient is a multiple of a dimension's size
 * belongs in front of that dimension, so only what is left over needs the
 * division and the modulo - usually a single loop variable instead of the
 * whole offset.
 */
inline IndexMap split_atoms(const IndexExpr &expr, const size_t *shape,
							int dims, IndexCode &state,
							bool free_only = false) {
	IndexMap result;
	result.expr_per_dim.resize(dims);
	IndexExpr rem = expr; // the part that divides, for the dimensions in front
	IndexExpr left;		  // the part that does not, masked per dimension
	long stride = 0;	  // what `left` is divided by, 0 while there is none
	long bias = 0;		  // what `left` was shifted by to stay positive
	unsigned int depth = 0;
	std::string value;
	// everything `left` is divided by, so that a shift by a multiple of it
	// changes none of the coordinates
	long total = 1;
	for (int i = 1; i < dims; i++)
		total *= (long)shape[i];
	for (int i = dims - 1; i > 0; i--) {
		const long size = (long)shape[i];
		// a dimension of size one is always indexed with zero
		if (size == 1) {
			result.expr_per_dim[i].a = 0;
			result.expr_per_dim[i].derive_bound(state.index_atoms);
			continue;
		}
		IndexExpr coord, higher; // rem = higher * size + coord
		coord.a = rem.a % size;
		higher.a = rem.a / size;
		for (const auto &[c, atom] : rem.exprs)
			if (c % size == 0)
				higher.exprs.push_back({c / size, atom});
			else
				coord.exprs.push_back({c, atom});
		coord.derive_bound(state.index_atoms);
		higher.derive_bound(state.index_atoms);
		rem = higher;
		if (!stride && coord.bound.first >= 0 && coord.bound.second < size) {
			// what does not divide stays inside its dimension, so the
			// decomposition is proven and needs no code
			result.expr_per_dim[i] = coord;
			continue;
		}
		// the caller may only want the decomposition if it is free
		if (free_only)
			return {};
		if (!stride) {
			// from here on the coordinates are masked out of `left`. A
			// negative value would let the divisions round towards zero, so
			// it is shifted by a multiple of everything it is divided by,
			// which the modulos do not see
			left = coord;
			bias = left.bound.first < 0
					   ? ((-left.bound.first + total - 1) / total) * total
					   : 0;
			left.a += bias;
			left.derive_bound(state.index_atoms);
			// it belongs into the scope of the atoms it is calculated from,
			// which is what takes it out of a loop it does not depend on
			depth = state.depth_of(left);
			value = state.to_code(left);
			stride = 1;
			coord = {};
		}
		// `left`s share of this dimension, if it reaches into it at all
		IndexExpr in_dim = coord;
		if (left.bound.second >= stride) {
			const std::string div =
				stride > 1 ? "(" + value + ") / " + std::to_string(stride)
						   : value;
			const int atom = state.declare_coord(
				depth, IndexCode::type_of({0, size - 1}),
				"(" + div + ") % " + std::to_string(size), 0, size - 1);
			in_dim.exprs.push_back({1, atom});
			// only a coordinate that is nothing but this one can be folded
			// back into the expression it was masked out of
			if (coord.a == 0 && coord.exprs.empty())
				state.split_coords[atom] = {left, stride, size};
		}
		in_dim.derive_bound(state.index_atoms);
		result.expr_per_dim[i] = in_dim;
		stride *= size;
	}
	// the outermost dimension holds everything that is left, but only if it
	// stays inside of it when nothing may be calculated
	if (free_only && dims > 0 &&
		(rem.bound.first < 0 || rem.bound.second >= (long)shape[0]))
		return {};
	if (dims > 0) {
		if (stride && left.bound.second >= stride) {
			// `left` reaches past the dimensions that masked it
			rem.exprs.push_back(
				{1,
				 state.declare_coord(
					 depth, IndexCode::type_of({0, left.bound.second / stride}),
					 "(" + value + ") / " + std::to_string(stride), 0,
					 left.bound.second / stride)});
			rem.derive_bound(state.index_atoms);
		}
		// the shift is a multiple of everything `left` was divided by, so
		// the dimension that holds the rest gives it back
		rem.a -= bias / (stride ? stride : 1);
		rem.derive_bound(state.index_atoms);
		result.expr_per_dim[0] = rem;
	}
	return result;
}
/** The flat element offset of the node that is being generated */
inline std::string node_offset(const FGraphNode *node, IndexCode &state) {
	return state.to_code(state.flatten(state.index_map, node->operation.shape,
									   node->operation.dimensions));
}
