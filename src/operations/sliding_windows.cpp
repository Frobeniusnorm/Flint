#include "sliding_windows.hpp"
#include "src/operations/implementation.hpp"

template <typename T>
void SlidingWindowImpl::unary_expression(T *__restrict__ result,
										 const T *__restrict__ data,
										 size_t from, size_t size,
										 const FGraphNode *curr) {
	const FOperation pred = curr->predecessors[0]->operation;
	const FSlidingWindow *slidewin =
		(FSlidingWindow *)curr->operation.additional_data;
	size_t acc_size = curr->operation.shape[1];
	std::vector<size_t> acc_sizes_pred =
		calcAccSizes(pred.dimensions, pred.shape);
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
		result[i] = data[base + offset];
	}
}
void SlidingWindowImpl::execute_cpu(const FGraphNode *node,
									std::vector<CPUResultData> predecessor_data,
									void *__restrict__ result, size_t from,
									size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
template <typename T>
void UnslideWindowImpl::unary_expression(T *__restrict__ result,
										 const T *__restrict__ data,
										 size_t from, size_t size,
										 const FGraphNode *curr) {
	const FOperation pred = curr->predecessors[0]->operation;
	const unsigned int *steps = (unsigned int *)curr->operation.additional_data;
	const std::vector<size_t> acc_sizes =
		calcAccSizes(curr->operation.dimensions, curr->operation.shape);
	const std::vector<size_t> acc_sizes_pred =
		calcAccSizes(pred.dimensions, pred.shape);
	size_t no_windows[pred.dimensions - 1];
	for (int i = 0; i < pred.dimensions - 1; i++) {
		size_t window_size = curr->operation.shape[i] - pred.shape[i + 1] + 1;
		no_windows[i] = window_size % steps[i] == 0
							? window_size / steps[i]
							: window_size / steps[i] + 1;
	}
	const std::vector<size_t> acc_no_windows =
		calcAccSizes(pred.dimensions - 1, no_windows);
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
				result[i] += data[wi + w * acc_sizes_pred[0]];
				wpp = 1;
			}
			w += wpp;
		}
	}
}
void UnslideWindowImpl::execute_cpu(const FGraphNode *node,
									std::vector<CPUResultData> predecessor_data,
									void *__restrict__ result, size_t from,
									size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
