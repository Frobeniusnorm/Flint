#include "sliding_windows.hpp"
#include "../utils.hpp"

using namespace std;

FGraphNode *SlidingWindowImpl::local_gradient(FGraphNode *y, int dx_i,
											  FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	if (0 == dx_i) {
		FSlidingWindow *sliding_win =
			(FSlidingWindow *)y->operation.additional_data;
		FGraphNode *res =
			funslide_window(prev_adj, a->operation.shape, sliding_win->step);
		return res;
	} else
		return nullptr;
}
template <typename T>
void SlidingWindowImpl::unary_expression(T *__restrict__ result,
										 const T *__restrict__ data,
										 size_t from, size_t size,
										 const FGraphNode *curr) {
	const FOperation pred = curr->predecessors[0]->operation;
	size_t total_el_size = 1;
	if (pred.op_type != FGEN_CONSTANT)
		for (int i = 0; i < pred.dimensions; i++)
			total_el_size *= pred.shape[i];
	const FSlidingWindow *slidewin =
		(FSlidingWindow *)curr->operation.additional_data;
	size_t acc_size = curr->operation.shape[1];
	std::vector<size_t> acc_sizes_pred =
		calc_acc_sizes(pred.dimensions, pred.shape);
	std::vector<size_t> acc_sizes_win(pred.dimensions);
	std::vector<size_t> acc_sizes_rest(pred.dimensions);
	acc_sizes_win[acc_sizes_win.size() - 1] = 1;
	acc_sizes_rest[acc_sizes_win.size() - 1] = 1;
	for (int i = acc_sizes_pred.size() - 2; i >= 0; i--) {
		acc_size *= curr->operation.shape[i + 2];
		acc_sizes_rest[i] = acc_sizes_rest[i + 1] * slidewin->size[i + 1];
		// no of windows in that dimension
		size_t window_size = pred.shape[i + 1] - slidewin->size[i + 1] + 1;
		window_size = window_size % slidewin->step[i + 1] == 0
						  ? window_size / slidewin->step[i + 1]
						  : window_size / slidewin->step[i + 1] + 1;
		acc_sizes_win[i] = acc_sizes_win[i + 1] * window_size;
	}
	for (size_t i = from; i < from + size; i++) {
		// window number
		size_t wi = i / acc_size;
		size_t rest = i % acc_size;
		// calculate window base from wi
		size_t base = 0;
		// index per dimension inside window from rest
		size_t offset = 0;
		for (int d = 0; d < pred.dimensions; d++) {
			size_t local_wi = wi / acc_sizes_win[d];
			// top left corner of window in that dimension
			size_t loc_base = local_wi * slidewin->step[d];
			base += loc_base * acc_sizes_pred[d];
			// remove this dimension from wi
			wi %= acc_sizes_win[d];
			size_t local_ri = rest / acc_sizes_rest[d];
			offset += local_ri * acc_sizes_pred[d];
			rest %= acc_sizes_rest[d];
		}
		result[i] = data[(base + offset) % total_el_size];
	}
}
int SlidingWindowImpl::generate_ocl_lazy(const FGraphNode *node,
										 std::string name,
										 OCLLazyCodegenState &compiler_state) {
	// the coordinates of the parameter are mapped in `mutate_index`
	compiler_state.code.prepend(
		"const " + type_string(node->operation.data_type) + " " + name +
		" = v" + to_string(compiler_state.variable_index + 1) + ";\n");
	return 0;
}
bool SlidingWindowImpl::mutate_index(const FGraphNode *node,
									 OCLLazyCodegenState &state) {
	const FOperation pred = node->predecessors[0]->operation;
	const FSlidingWindow *slidewin =
		(FSlidingWindow *)node->operation.additional_data;
	// the first dimension of the node enumerates the windows, the number of
	// windows per dimension is the shape that coordinate is addressed with
	vector<size_t> no_windows(pred.dimensions);
	for (int d = 0; d < pred.dimensions; d++) {
		const size_t span = pred.shape[d] - slidewin->size[d] + 1;
		no_windows[d] = span % slidewin->step[d] == 0
							? span / slidewin->step[d]
							: span / slidewin->step[d] + 1;
	}
	// the windows arrive as one coordinate, so splitting it into the position
	// per dimension is the one division this operation cannot avoid
	IndexMap windows = split_atoms(state.index_map.expr_per_dim[0],
								   no_windows.data(), pred.dimensions, state);
	IndexMap &pred_map = state.pred_index_maps.emplace_back();
	pred_map.current_depth = state.index_map.current_depth;
	pred_map.expr_per_dim.resize(pred.dimensions);
	for (int d = 0; d < pred.dimensions; d++) {
		// the windows are placed `step` entries apart, the rest is the
		// coordinate inside of the window, which the node already has
		IndexExpr &expr = pred_map.expr_per_dim[d];
		const IndexExpr &window = windows.expr_per_dim[d];
		expr = state.index_map.expr_per_dim[d + 1];
		expr.a += window.a * (long)slidewin->step[d];
		for (const auto &[c, atom] : window.exprs)
			expr.exprs.push_back({c * (long)slidewin->step[d], atom});
		expr.derive_bound(state.index_atoms);
	}
	return true;
}
void SlidingWindowImpl::execute_cpu(const FGraphNode *node,
									std::vector<CPUResultData> predecessor_data,
									void *__restrict__ result, size_t from,
									size_t size){UNARY_EXECUTE_MONOTON_IMPL}

FGraphNode *UnslideWindowImpl::local_gradient(FGraphNode *y, int dx_i,
											  FGraphNode *prev_adj) {
	// TODO
	return nullptr;
}
template <typename T>
void UnslideWindowImpl::unary_expression(T *__restrict__ result,
										 const T *__restrict__ data,
										 size_t from, size_t size,
										 const FGraphNode *curr) {
	const FOperation pred = curr->predecessors[0]->operation;
	size_t total_el_size = 1;
	if (pred.op_type != FGEN_CONSTANT)
		for (int i = 0; i < pred.dimensions; i++)
			total_el_size *= pred.shape[i];
	const unsigned int *steps = (unsigned int *)curr->operation.additional_data;
	const std::vector<size_t> acc_sizes =
		calc_acc_sizes(curr->operation.dimensions, curr->operation.shape);
	const std::vector<size_t> acc_sizes_pred =
		calc_acc_sizes(pred.dimensions, pred.shape);
	size_t no_windows[pred.dimensions - 1];
	for (int i = 0; i < pred.dimensions - 1; i++) {
		size_t window_size = curr->operation.shape[i] - pred.shape[i + 1] + 1;
		no_windows[i] = window_size % steps[i] == 0
							? window_size / steps[i]
							: window_size / steps[i] + 1;
	}
	const std::vector<size_t> acc_no_windows =
		calc_acc_sizes(pred.dimensions - 1, no_windows);
	for (size_t i = from; i < from + size; i++) {
		result[i] = 0;
		size_t first_w = 0, last_w = 0;
		// calculate first and last hit
		for (int d = 0; d < curr->operation.dimensions; d++) {
			const unsigned int id =
				(i / acc_sizes[d]) % curr->operation.shape[d];
			// first hit is where the window overlaps with the element of
			// window size - 1 before this element (since the window reaches
			// to this element)
			const size_t wdf = (size_t)std::ceil(
				(std::max(0l, (long)id - (long)pred.shape[d + 1] + 1) /
				 (double)steps[d]));
			const size_t wfl = id / steps[d];
			first_w += wdf * acc_no_windows[d];
			last_w += wfl * acc_no_windows[d];
		}
		size_t w = first_w;
		while (w <= last_w) {
			// tests if this window is a hit or not, if not calculates the
			// distance to the next window
			bool contained = true;
			size_t wi = 0;
			size_t wpp = 0;
			for (int d = curr->operation.dimensions - 1; d >= 0; d--) {
				const unsigned int wd = (w / acc_no_windows[d]) % no_windows[d];
				const unsigned int w_start = wd * steps[d];
				const unsigned int id =
					(i / acc_sizes[d]) % curr->operation.shape[d];
				if (id >= w_start && id < w_start + pred.shape[d + 1]) {
					wi += (id - w_start) * acc_sizes_pred[d + 1];
				} else {
					contained = false;
					// we cant break yet -> advance to next window in
					// dimension
					wpp += acc_no_windows[d];
				}
			}
			if (contained) {
				result[i] += data[(wi + w * acc_sizes_pred[0]) % total_el_size];
				wpp = 1;
			}
			w += wpp;
		}
	}
}
/** Geometry of one dimension of an unslide */
struct UnslideDim {
		long size;	 // of the window
		long step;	 // between two windows
		long nw;	 // number of windows
		long k;		 // number of windows an element can be part of
		IndexExpr q; // coordinate / step: the last window that may contain it
		IndexExpr r; // coordinate % step: its offset in that window
		std::string j; // loop variable of the candidates, if there are some
		// the window and the offset in it of a candidate, and their code if
		// they are variables of their own
		IndexExpr w, offset;
		std::string w_code, offset_code;
		bool misses_low() const { return q.bound.first < k - 1; }
		bool misses_high() const { return q.bound.second >= nw; }
		bool misses_window() const {
			return r.bound.second + step * (k - 1) >= size;
		}
};
struct Unslide {
		std::vector<UnslideDim> dims;
		bool loop = false, guarded = false;
		unsigned int depth; // of the candidates
};
/** The atom called `name`, it is registered if there is none yet. */
static int unslide_atom(OCLLazyCodegenState &state, const string &name,
						long hi, unsigned int depth) {
	for (int i = state.index_atoms.size() - 1; i >= 0; i--)
		if (state.index_atoms[i].name == name)
			return i;
	return state.add_atom(name, 0, hi, depth);
}
// An element x is part of the windows w = x / step - j whose offset
// x % step + j * step is still inside of them, for the j below
// ceil(size / step). If the windows do not overlap that is at most one, so
// the unslide is only an index map, else the candidates are summed in a loop.
// `mutate_index` and `generate_ocl_lazy` both derive it, the atoms are named
// after the operation so that both find the same ones.
static Unslide unslide_of(const FGraphNode *node, OCLLazyCodegenState &state) {
	const FOperation pred = node->predecessors[0]->operation;
	const unsigned int *steps = (unsigned int *)node->operation.additional_data;
	const string prefix = "unslide" + to_string(state.generated_ops) + "_";
	Unslide res;
	res.dims.resize(node->operation.dimensions);
	for (int d = 0; d < node->operation.dimensions; d++) {
		UnslideDim &u = res.dims[d];
		u.size = (long)pred.shape[d + 1];
		u.step = steps[d];
		const long span = (long)node->operation.shape[d] - u.size + 1;
		u.nw = (span + u.step - 1) / u.step;
		u.k = (u.size + u.step - 1) / u.step;
		const IndexExpr &x = state.index_map.expr_per_dim[d];
		if (u.step == 1) {
			u.q = x;
			u.r.derive_bound(state.index_atoms);
		} else {
			const size_t split[2] = {
				(node->operation.shape[d] + u.step - 1) / u.step,
				(size_t)u.step};
			IndexMap qr = split_atoms(x, split, 2, state);
			u.q = qr.expr_per_dim[0];
			u.r = qr.expr_per_dim[1];
		}
		res.loop |= u.k > 1;
		res.guarded |= u.misses_low() || u.misses_high() || u.misses_window();
	}
	res.depth = state.index_map.current_depth + res.loop;
	for (int d = 0; d < node->operation.dimensions; d++) {
		UnslideDim &u = res.dims[d];
		u.w = u.q;
		u.offset = u.r;
		string minus_j, plus_j;
		if (u.k > 1) {
			u.j = prefix + to_string(d);
			const int j = unslide_atom(state, u.j, u.k - 1, res.depth);
			u.w.exprs.push_back({-1, j});
			u.offset.exprs.push_back({u.step, j});
			minus_j = " - " + u.j;
			plus_j = " + " + to_string(u.step) + " * " + u.j;
		}
		// a candidate that misses is guarded away, so what is read lies
		// inside of the window. That bound is only true behind the guard,
		// as a variable of its own it holds for every use of the coordinate
		const auto inside = [&](IndexExpr &expr, string &code,
								const string &name, const string &value,
								long hi) {
			expr.derive_bound(state.index_atoms);
			if (!res.guarded ||
				(expr.bound.first >= 0 && expr.bound.second <= hi))
				return;
			code = value;
			expr.a = 0;
			expr.exprs = {{1, unslide_atom(state, name, hi, res.depth)}};
			expr.derive_bound(state.index_atoms);
		};
		inside(u.w, u.w_code, prefix + "w" + to_string(d),
			   state.to_code(u.q) + minus_j, u.nw - 1);
		inside(u.offset, u.offset_code, prefix + "o" + to_string(d),
			   state.to_code(u.r) + plus_j, u.size - 1);
	}
	return res;
}
bool UnslideWindowImpl::mutate_index(const FGraphNode *node,
									 OCLLazyCodegenState &state) {
	const FOperation pred = node->predecessors[0]->operation;
	const Unslide unslide = unslide_of(node, state);
	IndexMap &pred_map = state.pred_index_maps.emplace_back();
	pred_map.current_depth = unslide.depth;
	pred_map.expr_per_dim.resize(pred.dimensions);
	IndexExpr &window = pred_map.expr_per_dim[0];
	window.a = 0;
	long acc = 1; // windows of the dimensions behind
	for (int d = node->operation.dimensions - 1; d >= 0; d--) {
		const UnslideDim &u = unslide.dims[d];
		window.a += u.w.a * acc;
		for (const auto &[c, atom] : u.w.exprs)
			window.exprs.push_back({c * acc, atom});
		pred_map.expr_per_dim[d + 1] = u.offset;
		acc *= u.nw;
	}
	window.derive_bound(state.index_atoms);
	return true;
}
int UnslideWindowImpl::generate_ocl_lazy(const FGraphNode *node,
										 std::string name,
										 OCLLazyCodegenState &compiler_state) {
	const string type = type_string(node->operation.data_type);
	const string par = "v" + to_string(compiler_state.variable_index + 1);
	const Unslide unslide = unslide_of(node, compiler_state);
	string loops, cond, coords;
	const auto keep_if = [&cond](const string &c) {
		cond += (cond.empty() ? "" : " && ") + c;
	};
	for (const UnslideDim &u : unslide.dims) {
		if (!u.j.empty())
			loops += "for(int " + u.j + " = 0; " + u.j + " < " +
					 to_string(u.k) + "; " + u.j + "++)\n";
		const string q = compiler_state.to_code(u.q);
		const string r = compiler_state.to_code(u.r);
		if (u.misses_low())
			keep_if(u.j.empty() ? q + " >= 0" : u.j + " <= " + q);
		if (u.misses_high())
			keep_if(q + " < " + to_string(u.nw) +
					(u.j.empty() ? "" : " + " + u.j));
		if (u.misses_window())
			keep_if(r +
					(u.j.empty() ? ""
								 : " + " + to_string(u.step) + " * " + u.j) +
					" < " + to_string(u.size));
		// declared behind the guard, which is what makes their bounds true
		for (const auto &[expr, code] :
			 {std::pair{&u.w, &u.w_code}, {&u.offset, &u.offset_code}})
			if (!code->empty()) {
				const IndexAtom &atom =
					compiler_state.index_atoms[expr->exprs[0].second];
				coords += compiler_state.type_of(atom.bound) + " " +
						  atom.name + " = " + *code + ";\n";
			}
	}
	if (!unslide.loop) {
		if (cond.empty())
			compiler_state.code.prepend("const " + type + " " + name + " = " +
										par + ";\n");
		else
			compiler_state.pred_wrappers = {
				{type + " " + name + " = 0;\nif(" + cond + "){\n" + coords,
				 name + " = " + par + ";\n}\n"}};
		return 0;
	}
	compiler_state.insert_code(type + " " + name + " = 0;\n" + loops + "{\n",
							   true);
	if (cond.empty())
		compiler_state.code.prepend(name + " += " + par + ";\n}\n");
	else {
		compiler_state.pred_wrappers = {{"if(" + cond + "){\n" + coords,
										 name + " += " + par + ";\n}\n"}};
		compiler_state.code.prepend("}\n");
	}
	return 0;
}
void UnslideWindowImpl::execute_cpu(const FGraphNode *node,
									std::vector<CPUResultData> predecessor_data,
									void *__restrict__ result, size_t from,
									size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
