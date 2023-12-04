#include "convolution.hpp"
#include "src/operations/implementation.hpp"

#define MIN_VAL(x, y) (x < y ? x : y)
#define MAX_VAL(x, y) (x < y ? y : x)

template <typename T, typename A, typename B>
void ConvolveImpl::binary_expression(T *__restrict__ result,
									 const A *__restrict__ data1,
									 const B *__restrict__ data2, size_t from,
									 size_t size, size_t index_man_1,
									 size_t inv_man_1, size_t index_man_2,
									 size_t inv_man_2, const FGraphNode *curr) {
	const FOperation op = curr->operation;
	const FGraphNode *gnp1 = curr->predecessors[0],
					 *gnp2 = curr->predecessors[1];
	const FOperation pred = gnp1->operation, kernel = gnp2->operation;
	const unsigned int *steps = (unsigned int *)op.additional_data;
	const bool multiple_filter =
		gnp2->operation.dimensions != gnp1->operation.dimensions;
	// calculate accumulated sizes for result, kernel and source (pred)
	std::vector<size_t> acc_sizes = calcAccSizes(op);
	std::vector<size_t> acc_sizes_pred = calcAccSizes(pred);
	std::vector<size_t> acc_sizes_kernel = calcAccSizes(kernel);
	size_t kernel_num_elems = kernel.shape[acc_sizes.size()];
	size_t pred_num_elems = multiple_filter ? 1 : pred.shape[acc_sizes.size()];
	for (long d = acc_sizes.size() - 1; d >= 0; d--) {
		pred_num_elems *= pred.shape[d];
		if (d != 0 || !multiple_filter) // since kernel.shape[0] is the
										// dimension of filters
			kernel_num_elems *= kernel.shape[d];
	}
	for (size_t i = from; i < from + size; i++) {
		size_t j = 0;
		// we can ignore last index of source and kernel for result since we
		// iterate over it (i.e. for the destination it is 0 since it does
		// not have that dimension)
		for (unsigned int d = 0;
			 d < (multiple_filter ? op.dimensions - 1 : op.dimensions); d++) {
			// get dimension index
			size_t di = (d == 0 ? i : i % acc_sizes[d - 1]) / acc_sizes[d];
			// reproject
			j += di * steps[d] * acc_sizes_pred[d];
		}
		// we must offset the kernel by the filter index, which is the last
		// dimension of the result
		size_t kernel_offset = 0;
		if (multiple_filter) {
			// filter index
			size_t fi = (i % acc_sizes[op.dimensions - 2]) /
						acc_sizes[op.dimensions - 1];
			kernel_offset = fi * kernel_num_elems; // since the filters are the
		}
		// now that we have the correct base index in source, convolve
		T res = 0;
		for (size_t k = 0; k < kernel_num_elems; k++) {
			bool set_zero = false;
			size_t o = 0; // source offset
			// reproject kernel
			const unsigned int last_dim = multiple_filter
											  ? acc_sizes_kernel.size() - 1
											  : acc_sizes_kernel.size();
			for (unsigned int d = 0; d < last_dim; d++) {
				const unsigned int kn_d = multiple_filter ? d + 1 : d;
				size_t di = 0;
				if (d != last_dim - 1)
					di = (d == 0 ? i : i % acc_sizes[d - 1]) / acc_sizes[d];
				size_t dk = (kn_d == 0 ? k : k % acc_sizes_kernel[kn_d - 1]) /
							acc_sizes_kernel[kn_d];
				if (d < pred.dimensions - 1)
					if (((di * steps[d]) + dk) * acc_sizes_pred[d] >=
							pred_num_elems ||
						(d > 0 && ((di * steps[d]) + dk) * acc_sizes_pred[d] >=
									  acc_sizes_pred[d - 1])) {
						set_zero = true;
						break;
					}
				o += dk * acc_sizes_pred[d];
			}
			if (set_zero)
				continue;
			res += data2[k + kernel_offset] * data1[j + o];
		}
		result[i] = res;
	}
}
template <typename T, typename A, typename B>
void GradientConvolve1Impl::binary_expression(
	T *__restrict__ result, const A *__restrict__ data1,
	const B *__restrict__ data2, size_t from, size_t size, size_t index_man_1,
	size_t inv_man_1, size_t index_man_2, size_t inv_man_2,
	const FGraphNode *curr) {
	/* This one is complicated so here is a quick explanation:
	 * The complicated part is that each value of the original image may
	 * overlap with several kernel multiplications, each of them
	 * corresponding to one window in the adjoint. So we iterate per image
	 * pixel over those overlapping windows. We can precalculate how many
	 * max. overlap one element, but that leads to the problem, that
	 * windows, which are not possible for the element will be counted too,
	 * which don't exist in the adjacent, so we have to skip those windows
	 * if no other window has already been counted (since if one window has
	 * been counted, the overlapping impossible windows are needed for the
	 * dimensional projection). If there is still a bug in this procedure
	 * bless your poor soul that has to fix it. Maybe rewriting it is
	 * smarter. */
	const FOperation op = curr->operation;
	const FGraphNode *gnp1 = curr->predecessors[0],
					 *gnp2 = curr->predecessors[1];
	const FOperation kernel = gnp1->operation, a = gnp2->operation;
	const unsigned int *steps = (unsigned int *)op.additional_data;
	// calculate accumulated sizes for result (pred), kernel and a
	// (adjacent)
	std::vector<size_t> acc_sizes = calcAccSizes(a);
	std::vector<size_t> acc_sizes_pred = calcAccSizes(op);
	std::vector<size_t> acc_sizes_kernel = calcAccSizes(kernel);
	acc_sizes[op.dimensions - 2] = 1;
	// accumulations of overlapping elements (kernel overlapping itself)
	std::vector<size_t> acc_overlapping(op.dimensions - 1);
	acc_overlapping[acc_overlapping.size() - 1] = 1;
	for (int i = acc_overlapping.size() - 2; i >= 0; i--) {
		acc_overlapping[i] =
			MAX_VAL(1, (long)std::ceil((double)kernel.shape[i + 1] /
									   (double)steps[i + 1])) *
			acc_overlapping[i + 1];
	}
	// First dimension overlap
	const size_t overlapping =
		MAX_VAL(1,
				(long)std::ceil((double)kernel.shape[0] / (double)steps[0])) *
		acc_overlapping[0];
	for (size_t i = from; i < from + size; i++) {
		T res = 0;
		bool in_steps = true;
		bool started_counting = false;
		// get base indices
		size_t keri = 0;
		size_t adji = 0;
		for (int d = 0; d < op.dimensions - 1; d++) {
			const size_t di =
				(d == 0 ? i : i % acc_sizes_pred[d - 1]) / acc_sizes_pred[d];
			// first kernel element is the offset from di to the first
			// kernel that overlaps it
			const size_t ki = di - (di / steps[d]) * steps[d];
			// if this index is outside the kernel size -> i is not
			// overlapped by a kernel
			if (ki >= kernel.shape[d]) {
				in_steps = false;
				break;
			}
			// first window for this index
			const size_t wdf = (size_t)std::ceil(
				(std::max(0l, (long)di - (long)kernel.shape[d] + 1) /
				 (double)steps[d]));
			keri += ki * acc_sizes_kernel[d];
			adji += wdf * acc_sizes[d];
		}
		if (in_steps) {
			// kernel offset for last index
			keri += i % op.shape[op.dimensions - 1];
			size_t actual_overlapping = 0;
			// iterate over overlapping windows = elements in a
			for (size_t o = 0; o < overlapping; o++) {
				// offsets
				size_t adjo = 0;
				size_t kero = 0;
				bool skip_kernel = false;
				for (int d = 0; d < op.dimensions - 1; d++) {
					// for each index adji will point to the first window in
					// that dimension calculate overlap in each dimension
					// and add it to the adjacent offset
					const size_t di = (d == 0 ? i : i % acc_sizes_pred[d - 1]) /
									  acc_sizes_pred[d];
					const size_t io =
						(d == 0 ? o : o % acc_overlapping[d - 1]) /
						acc_overlapping[d];
					const size_t ao =
						(d == 0 ? actual_overlapping
								: actual_overlapping % acc_overlapping[d - 1]) /
						acc_overlapping[d];
					// check if kernel offset is feasible (the kernel we
					// take the offset to is in bounds)
					const size_t ki =
						(d == 0 ? keri : keri % acc_sizes_kernel[d - 1]) /
						acc_sizes_kernel[d];
					if (di + kernel.shape[d] - (ki + io * steps[d]) >
						op.shape[d]) {
						// those cases are no real windows, only skip them
						// if there haven't been real windows yet
						if (!started_counting) {
							actual_overlapping--;
						}
						skip_kernel = true;
						break;
					} else if (ki + io * steps[d] >= kernel.shape[d] ||
							   di < ki + io * steps[d]) {
						skip_kernel = true;
						break;
					}
					adjo += ao * acc_sizes[d];
					kero += io * steps[d] * acc_sizes_kernel[d];
				}
				if (!skip_kernel) {
					started_counting = true;
					res += data1[keri + kero] * data2[adjo + adji];
				}
				actual_overlapping++;
			}
		}
		result[i] = res;
	}
}
template <typename T, typename A, typename B>
void GradientConvolve2Impl::binary_expression(
	T *__restrict__ result, const A *__restrict__ data1,
	const B *__restrict__ data2, size_t from, size_t size, size_t index_man_1,
	size_t inv_man_1, size_t index_man_2, size_t inv_man_2,
	const FGraphNode *curr) {
	// normal convolution:
	//   shape(op) = [k1, k2, ..., kn, c]
	//   shape(pred) = [p1, p2, ..., pn, c]
	//   shape(prev_adj) = [w1, w2, ..., wn]
	// multifilter convolution:
	//   shape(op) = [filter, k1, k2, ..., kn, c]
	//   shape(pred) = [p1, p2, ..., pn, c]
	//   shape(prev_adj) = [w1, w2, ..., wn, filter]
	const FOperation op = curr->operation;
	const FGraphNode *gnp1 = curr->predecessors[0],
					 *gnp2 = curr->predecessors[1];
	const FOperation pred = gnp1->operation, prev_adj = gnp2->operation;
	const std::vector<size_t> acc_sizes_pred = calcAccSizes(pred);
	const std::vector<size_t> acc_sizes_kernel = calcAccSizes(op);
	const bool multifilter = op.dimensions > pred.dimensions;
	// like accumulated sizes for prev_adj but without filter in multifilter
	// context
	std::vector<size_t> acc_sizes_windows(multifilter ? prev_adj.dimensions - 1
													  : prev_adj.dimensions);
	acc_sizes_windows[acc_sizes_windows.size() - 1] = 1;
	for (int i = acc_sizes_windows.size() - 2; i >= 0; i--) {
		acc_sizes_windows[i] = acc_sizes_windows[i + 1] * prev_adj.shape[i + 1];
	}
	// total number of windows
	const size_t windows = acc_sizes_windows[0] * prev_adj.shape[0];
	// helper variables
	const size_t num_elems_kernel =
		multifilter ? acc_sizes_kernel[0] : acc_sizes_kernel[0] * op.shape[0];
	const unsigned int *steps = (unsigned int *)op.additional_data;
	const unsigned int num_filter = multifilter ? op.shape[0] : 1;
	for (size_t i = from; i < from + size; i++) {
		// filter entry of current iteration for multifilter
		size_t f = 0;
		if (multifilter) {
			f = i / num_elems_kernel;
		}
		// project kernel offset to a offset
		size_t a_offset = 0;
		for (int j = multifilter ? 1 : 0; j < op.dimensions; j++) {
			size_t ki = (i / acc_sizes_kernel[j]) % op.shape[j];
			a_offset += ki * acc_sizes_pred[multifilter ? j - 1 : j];
		}
		result[i] = 0;
		// iterate over windows = adjoint elements in first dimensions
		for (size_t w = 0; w < windows; w++) {
			// calculate start value of window for pred
			size_t a = 0;
			for (int j = 0; j < acc_sizes_windows.size(); j++) {
				size_t wj = (w / acc_sizes_windows[j]) % prev_adj.shape[j];
				a += wj * acc_sizes_pred[j] * steps[j];
			}
			result[i] += data1[a + a_offset] * data2[w * num_filter + f];
		}
	}
}
void ConvolveImpl::execute_cpu(const FGraphNode *node,
							   std::vector<CPUResultData> predecessor_data,
							   void *__restrict__ result, size_t from,
							   size_t size) {
	BINARY_EXECUTE_IMPL
}
void GradientConvolve1Impl::execute_cpu(
	const FGraphNode *node, std::vector<CPUResultData> predecessor_data,
	void *__restrict__ result, size_t from, size_t size) {
	BINARY_EXECUTE_IMPL
}
void GradientConvolve2Impl::execute_cpu(
	const FGraphNode *node, std::vector<CPUResultData> predecessor_data,
	void *__restrict__ result, size_t from, size_t size) {
	BINARY_EXECUTE_IMPL
}
