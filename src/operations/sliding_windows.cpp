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
int UnslideWindowImpl::generate_ocl_lazy(const FGraphNode *node,
										 std::string name,
										 OCLLazyCodegenState &compiler_state) {

	FGraphNode *gnp1 = node->predecessors[0];
	const FOperation pred = gnp1->operation;
	size_t total_el_size = 1;
	if (pred.op_type != FGEN_CONSTANT)
		for (int i = 0; i < pred.dimensions; i++)
			total_el_size *= pred.shape[i];
	const string par1 = compiler_state.findOrInsertParameter(gnp1);
	const unsigned int *steps = (unsigned int *)node->operation.additional_data;
	const vector<size_t> acc_sizes =
		calc_acc_sizes(node->operation.dimensions, node->operation.shape);
	const vector<size_t> acc_sizes_pred =
		calc_acc_sizes(pred.dimensions, pred.shape);
	size_t no_windows[pred.dimensions - 1];
	for (int i = 0; i < pred.dimensions - 1; i++) {
		size_t window_size = node->operation.shape[i] - pred.shape[i + 1] + 1;
		no_windows[i] = window_size % steps[i] == 0
							? window_size / steps[i]
							: window_size / steps[i] + 1;
	}
	const vector<size_t> acc_no_windows =
		calc_acc_sizes(pred.dimensions - 1, no_windows);
	// which windows an element is part of follows from its coordinates
	IndexMap coords = compiler_state.index_map;
	vector<string> coord(node->operation.dimensions);
	for (int d = 0; d < node->operation.dimensions; d++)
		coord[d] = compiler_state.to_code(coords.expr_per_dim[d]);
	Twine local_code = type_string(node->operation.data_type) + " " + name +
					   " = 0;\n"
					   "{\n"
					   "const " +
					   compiler_state.index_type + " first_w = 0";
	for (int d = node->operation.dimensions - 1; d >= 0; d--) {
		local_code += " + max((" + compiler_state.index_type + ")0, " +
					  coord[d] + " - " + to_string(pred.shape[d + 1]) +
					  " + 1) / " + to_string(steps[d]) + " * " +
					  to_string(acc_no_windows[d]);
	}
	local_code += ";\nconst " + compiler_state.index_type + " last_w = 0";
	for (int d = node->operation.dimensions - 1; d >= 0; d--) {
		local_code += " + " + coord[d] + " / " + to_string(steps[d]) + " * " +
					  to_string(acc_no_windows[d]);
	}
	local_code += ";\nfor(" + compiler_state.index_type +
				  " w=first_w;w<=last_w;){\n"
				  " bool contained = true;\n"
				  " " +
				  compiler_state.index_type +
				  " wi = 0;\n"
				  " " +
				  compiler_state.index_type + " wpp = 0;\n";
	for (int d = node->operation.dimensions - 1; d >= 0; d--) {
		local_code += " {\n"
					  "  const " +
					  compiler_state.index_type + " w_start=((w/" +
					  to_string(acc_no_windows[d]) + ")%" +
					  to_string(no_windows[d]) + ")*" + to_string(steps[d]) +
					  ";\n"
					  "  const " +
					  compiler_state.index_type + " id=" + coord[d] +
					  ";\n"
					  "  if(id>=w_start && id<w_start+" +
					  to_string(pred.shape[d + 1]) +
					  ")\n"
					  "   wi+=(id-w_start)*" +
					  to_string(acc_sizes_pred[d + 1]) +
					  ";\n"
					  "  else{\n"
					  "   contained = false;\n"
					  "   wpp += " +
					  to_string(acc_no_windows[d]) +
					  ";\n"
					  "  }\n"
					  " }\n";
	}
	local_code += " if(contained) {"
				  "  " +
				  name + "+=" + par1 + "[(wi+w*" +
				  to_string(acc_sizes_pred[0]) + ")%" +
				  to_string(total_el_size) +
				  "];\n"
				  "  wpp = 1;\n}\n"
				  " w += wpp;\n"
				  "}\n"
				  "}";
	compiler_state.code.prepend(local_code);
	return OCL_LAZY_DONT_PUSH_PREDS;
}
void UnslideWindowImpl::execute_cpu(const FGraphNode *node,
									std::vector<CPUResultData> predecessor_data,
									void *__restrict__ result, size_t from,
									size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
