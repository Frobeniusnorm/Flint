#include "convolution.hpp"
#include "../backend_ocl/utils.hpp"
#include "../utils.hpp"
#include "flint.h"
#include "flint_helper.hpp"

#define MIN_VAL(x, y) (x < y ? x : y)
#define MAX_VAL(x, y) (x < y ? y : x)

using namespace std;

FGraphNode *ConvolveImpl::gradient_convolve2(FGraphNode *a, FGraphNode *kernel,
											 FGraphNode *prev_adj,
											 const unsigned int *steps) {
	if (!kernel->result_data)
		fExecuteGraph(kernel);
	if (!prev_adj->result_data)
		fExecuteGraph(prev_adj);
	FGraphNode *gradient = new FGraphNode();
	gradient->num_predecessor = 2;
	gradient->predecessors = safe_mal<FGraphNode *>(2);
	if (!gradient->predecessors)
		return nullptr;
	gradient->predecessors[0] = a;
	gradient->predecessors[1] = prev_adj;
	a->reference_counter++;
	prev_adj->reference_counter++;
	gradient->result_data = nullptr;
	gradient->reference_counter = 0;
	FOperation op;
	op.broadcasting_mode = 0;
	op.data_type =
		higher_type(kernel->operation.data_type, prev_adj->operation.data_type);
	op.dimensions = kernel->operation.dimensions;
	op.shape = safe_mal<size_t>(op.dimensions);
	if (!op.shape)
		return nullptr;
	memcpy(op.shape, kernel->operation.shape, op.dimensions * sizeof(size_t));
	op.op_type = FGRADIENT_CONVOLVE2;
	op.additional_data = safe_mal<unsigned int>(a->operation.dimensions - 1);
	if (!op.additional_data)
		return nullptr;
	memcpy(op.additional_data, steps,
		   (a->operation.dimensions - 1) * sizeof(unsigned int));
	gradient->operation = op;
	OperationImplementation::configure_gradient_information(gradient,
															{a, prev_adj});
	return gradient;
}
FGraphNode *ConvolveImpl::gradient_convolve1(FGraphNode *a, FGraphNode *kernel,
											 FGraphNode *prev_adj,
											 const unsigned int *steps) {
	if (!kernel->result_data)
		fExecuteGraph(kernel);
	if (!prev_adj->result_data)
		fExecuteGraph(prev_adj);
	FGraphNode *gradient = new FGraphNode();
	gradient->num_predecessor = 2;
	gradient->predecessors = safe_mal<FGraphNode *>(2);
	if (!gradient->predecessors)
		return nullptr;
	gradient->predecessors[0] = kernel;
	gradient->predecessors[1] = prev_adj;
	kernel->reference_counter++;
	prev_adj->reference_counter++;
	gradient->result_data = nullptr;
	gradient->reference_counter = 0;
	FOperation op;
	op.broadcasting_mode = 0;
	op.data_type =
		higher_type(kernel->operation.data_type, prev_adj->operation.data_type);
	op.dimensions = a->operation.dimensions;
	op.shape = safe_mal<size_t>(op.dimensions);
	if (!op.shape)
		return nullptr;
	memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
	op.op_type = FGRADIENT_CONVOLVE1;
	op.additional_data = safe_mal<unsigned int>(a->operation.dimensions - 1);
	if (!op.additional_data)
		return nullptr;
	memcpy(op.additional_data, steps,
		   (a->operation.dimensions - 1) * sizeof(unsigned int));
	gradient->operation = op;
	OperationImplementation::configure_gradient_information(gradient,
															{kernel, prev_adj});
	return gradient;
}
FGraphNode *ConvolveImpl::local_gradient(FGraphNode *y, int dx_i,
										 FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	FGraphNode *kernel = y->predecessors[1];
	const unsigned int *steps = (unsigned int *)y->operation.additional_data;
	if (0 == dx_i) {
		return gradient_convolve1(a, kernel, prev_adj, steps);
	} else if (1 == dx_i) {
		if (y->operation.op_type == FCONVOLVE)
			return gradient_convolve2(a, kernel, prev_adj, steps);
	}
	return nullptr;
}
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
	// total sizes
	size_t num_entries1 = 1, num_entries2 = 1;
	if (gnp1->operation.op_type != FGEN_CONSTANT)
		for (int i = 0; i < gnp1->operation.dimensions; i++)
			num_entries1 *= gnp1->operation.shape[i];
	if (gnp2->operation.op_type != FGEN_CONSTANT)
		for (int i = 0; i < gnp2->operation.dimensions; i++)
			num_entries2 *= gnp2->operation.shape[i];
	// calculate accumulated sizes for result, kernel and source (pred)
	std::vector<size_t> acc_sizes = calc_acc_sizes(op);
	std::vector<size_t> acc_sizes_pred = calc_acc_sizes(pred);
	std::vector<size_t> acc_sizes_kernel = calc_acc_sizes(kernel);
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
			res += data2[(k + kernel_offset) % num_entries2] *
				   data1[(j + o) % num_entries1];
		}
		result[i] = res;
	}
}
int ConvolveImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
									OCLLazyCodegenState &compiler_state) {
	FGraphNode *gnp1 = node->predecessors[0], *gnp2 = node->predecessors[1];
	const string par1 = compiler_state.findOrInsertParameter(gnp1),
				 par2 = compiler_state.findOrInsertParameter(gnp2);
	const bool multiple_filter =
		gnp2->operation.dimensions != gnp1->operation.dimensions;
	const FOperation op = node->operation;
	const FOperation pred = gnp1->operation, kernel = gnp2->operation;
	const unsigned int *steps = (unsigned int *)op.additional_data;
	const vector<size_t> acc_sizes = calc_acc_sizes(op);
	const vector<size_t> acc_sizes_pred = calc_acc_sizes(pred);
	const vector<size_t> acc_sizes_kernel = calc_acc_sizes(kernel);
	size_t kernel_num_elems = kernel.shape[acc_sizes.size()];
	size_t pred_num_elems = multiple_filter ? 1 : pred.shape[acc_sizes.size()];
	for (long d = acc_sizes.size() - 1; d >= 0; d--) {
		pred_num_elems *= pred.shape[d];
		if (d != 0 || !multiple_filter) // since kernel.shape[0] is
										// the dimension of filters
			kernel_num_elems *= kernel.shape[d];
	}
	const std::string type = type_string(node->operation.data_type);
	Twine conv_code;
	conv_code += type + " " + name + " = 0;\n{\nlong j = 0";
	for (unsigned int d = 0;
		 d < (multiple_filter ? op.dimensions - 1 : op.dimensions); d++)
		conv_code += " + (" +
					 (d == 0 ? string("index")
							 : "index % " + to_string(acc_sizes[d - 1])) +
					 " / " + to_string(acc_sizes[d]) + ") * " +
					 to_string(steps[d] * acc_sizes_pred[d]);
	conv_code +=
		";\nlong kernel_offset = " +
		(multiple_filter
			 ? string("(index % " + to_string(acc_sizes[op.dimensions - 2]) +
					  ") / " + to_string(acc_sizes[op.dimensions - 1]) + " * " +
					  to_string(kernel_num_elems))
			 : string("0")) +
		";\n" + type_string(op.data_type) +
		" res = 0;\n"
		"for(long k = 0; k < " +
		to_string(kernel_num_elems) +
		"; k++){\n"
		" long o = 0;\n";
	const unsigned int last_dim =
		multiple_filter ? acc_sizes_kernel.size() - 1 : acc_sizes_kernel.size();
	for (unsigned int d = 0; d < last_dim; d++) {
		const unsigned int kn_d = multiple_filter ? d + 1 : d;
		conv_code +=
			"{\nconst long di = " +
			(d == last_dim - 1
				 ? "0"
				 : (d == 0 ? string("index")
						   : "index % " + to_string(acc_sizes[d - 1])) +
					   " / " + to_string(acc_sizes[d])) +
			";\n"
			"const long dk = " +
			(kn_d == 0 ? string("k")
					   : "k % " + to_string(acc_sizes_kernel[kn_d - 1])) +
			"/ " + to_string(acc_sizes_kernel[kn_d]) + ";\n";
		if (d < pred.dimensions - 1) {
			conv_code += "if((di * " + to_string(steps[d]) + " + dk) * " +
						 to_string(acc_sizes_pred[d]) +
						 " >= " + to_string(pred_num_elems);
			if (d > 0)
				conv_code += " || (di * " + to_string(steps[d]) + " + dk) * " +
							 to_string(acc_sizes_pred[d]) +
							 " >= " + to_string(acc_sizes_pred[d - 1]);
			conv_code += ") continue;\n";
		}
		conv_code += "o += dk * " + to_string(acc_sizes_pred[d]) + ";\n}\n";
	}
	string ind1 = "j + o";
	string ind2 = "k + kernel_offset";
	if (gnp1->operation.op_type == FGEN_CONSTANT)
		ind1 = "0";
	if (gnp2->operation.op_type == FGEN_CONSTANT)
		ind2 = "0";
	conv_code += "res += " + par2 + "[" + ind2 + "] * " + par1 + "[" + ind1 +
				 "];\n}\n" + name + " = res;\n}\n";
	compiler_state.code.prepend(conv_code);
	return OCL_LAZY_DONT_PUSH_PREDS;
}
std::string ConvolveImpl::generate_ocl_parameters_eager(
	FType res_type, std::vector<FType> parameter_types) {
	return ", const __global " + type_string(parameter_types[0]) +
		   "* P0"
		   ", const long num_entries0, const int dimensions0"
		   ", const __global " +
		   type_string(parameter_types[1]) +
		   "* P1"
		   ", const long num_entries1, const int dimensions1"
		   ", __constant long* acc_sizes, __constant long* acc_sizes_pred, "
		   "__constant long* acc_sizes_kernel"
		   ", __constant int* steps, long total_elements_image, long "
		   "total_elements_kernel";
}
std::string
ConvolveImpl::generate_ocl_eager(FType res_type,
								 std::vector<FType> parameter_types) {
	return "if(index >= num_entriesR) return;\n"
		   "int multi_filter = dimensions0 != dimensions1;\n"
		   "long j = 0;\n"
		   "for(int d = 0; d < dimensions0 - 1; d++){\n"
		   " long di = (d == 0 ? index : index % acc_sizes[d - 1]) / "
		   "acc_sizes[d];\n"
		   " j += di * steps[d] * acc_sizes_pred[d];\n"
		   "}\n"
		   "long kernel_offset = 0;\n"
		   "if(multi_filter){\n"
		   " long fi = (index % acc_sizes[dimensions0 - 2]) / "
		   "acc_sizes[dimensions0 - 1];\n"
		   " kernel_offset = fi * acc_sizes_kernel[0];\n"
		   "}\n" +
		   type_string(res_type) +
		   " res = 0;\n"
		   "const long kernel_num_elems = multi_filter ? acc_sizes_kernel[0] "
		   ": "
		   "total_elements_kernel;\n"
		   "for(long k = 0; k < kernel_num_elems; k++){\n"
		   " bool set_zero = false;\n"
		   " long o = 0;\n"
		   " const int last_dim = multi_filter ? dimensions1 - 1 : "
		   "dimensions1;\n"
		   " for(int d = 0; d < last_dim; d++){\n"
		   "  const int kn_d = multi_filter ? d + 1 : d;\n"
		   "  long di = d == last_dim ? 0 : (d == 0 ? index : index % "
		   "acc_sizes[d - 1]) / "
		   "acc_sizes[d];\n"
		   "  long dk = (kn_d == 0 ? k : k % acc_sizes_kernel[kn_d - 1]) / "
		   "acc_sizes_kernel[kn_d];\n"
		   "  if(d < dimensions0 - 1)\n"
		   "   if(((di * steps[d]) + dk) * acc_sizes_pred[d] >= "
		   "total_elements_image"
		   "||\n"
		   "        (d > 0 && ((di * steps[d]) + dk) * acc_sizes_pred[d] >= \n"
		   "acc_sizes_pred[d - 1])) {\n"
		   "    set_zero = true; break;\n}\n"
		   "  o += dk * acc_sizes_pred[d];\n"
		   " }\n"
		   " if (set_zero) continue;\n"
		   " res += P1[(k + kernel_offset) % num_entries1] * P0[(j + o) % "
		   "num_entries0];\n"
		   "}\n"
		   "R[index] = res;";
}
void ConvolveImpl::push_additional_kernel_parameters(
	FGraphNode *node, cl_kernel kernel, cl_context context, int &par_index,
	std::list<cl_mem> &to_free) {
	cl_int err_code;
	const FOperation op = node->operation;
	const FGraphNode *gnp1 = node->predecessors[0],
					 *gnp2 = node->predecessors[1];
	const FOperation pred = gnp1->operation, kernel_par = gnp2->operation;
	unsigned int *steps = (unsigned int *)op.additional_data;
	// allocate steps
	cl_mem steps_mem =
		clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					   op.dimensions * sizeof(int), steps, &err_code);
	if (!steps_mem)
		flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
							  std::to_string(err_code));
	to_free.push_back(calc_and_push_acc_size(op.dimensions, op.shape, kernel,
											 context, par_index));
	to_free.push_back(calc_and_push_acc_size(pred.dimensions, pred.shape,
											 kernel, context, par_index));
	to_free.push_back(calc_and_push_acc_size(
		kernel_par.dimensions, kernel_par.shape, kernel, context, par_index));
	if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem),
					   (void *)&steps_mem) != CL_SUCCESS)
		flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
							  std::to_string(err_code));
	to_free.push_back(steps_mem);
	// total size of image (because of constants that have size of 1 in result)
	size_t total_elements_image = 1, total_elements_kernel = 1;
	for (int i = 0; i < gnp1->operation.dimensions; i++)
		total_elements_image *= gnp1->operation.shape[i];
	for (int i = 0; i < gnp2->operation.dimensions; i++)
		total_elements_kernel *= gnp2->operation.shape[i];
	if (clSetKernelArg(kernel, par_index++, sizeof(size_t),
					   (void *)&total_elements_image) != CL_SUCCESS)
		flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
							  std::to_string(err_code));
	if (clSetKernelArg(kernel, par_index++, sizeof(size_t),
					   (void *)&total_elements_kernel) != CL_SUCCESS)
		flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
							  std::to_string(err_code));
}
FGraphNode *GradientConvolve1Impl::local_gradient(FGraphNode *y, int dx_i,
												  FGraphNode *prev_adj) {
	FGraphNode *kernel = y->predecessors[0];
	FGraphNode *a = y->predecessors[1];
	if (1 == dx_i) {
		const unsigned int *steps =
			(unsigned int *)y->operation.additional_data;
		if (!kernel->result_data)
			fExecuteGraph(kernel);
		FGraphNode *gradient = new FGraphNode();
		gradient->num_predecessor = 2;
		gradient->predecessors = safe_mal<FGraphNode *>(2);
		if (!gradient->predecessors)
			return nullptr;
		gradient->predecessors[0] = kernel;
		gradient->predecessors[1] = prev_adj;
		kernel->reference_counter++;
		prev_adj->reference_counter++;
		gradient->result_data = nullptr;
		gradient->reference_counter = 0;
		FOperation op;
		op.data_type = higher_type(kernel->operation.data_type,
								   prev_adj->operation.data_type);
		op.dimensions = a->operation.dimensions;
		op.shape = safe_mal<size_t>(op.dimensions);
		if (!op.shape)
			return nullptr;
		memcpy(op.shape, a->operation.shape, op.dimensions * sizeof(size_t));
		op.op_type = FGRADIENT_CONVOLVE1;
		op.additional_data =
			safe_mal<unsigned int>(a->operation.dimensions - 1);
		if (!op.additional_data)
			return nullptr;
		memcpy(op.additional_data, steps,
			   (a->operation.dimensions - 1) * sizeof(unsigned int));
		gradient->operation = op;
		OperationImplementation::configure_gradient_information(
			gradient, {kernel, prev_adj});
		return gradient;
	} else if (0 == dx_i) {
		return ConvolveImpl::gradient_convolve1(
			prev_adj, kernel, a, (unsigned int *)y->operation.additional_data);
	}
	return nullptr;
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
	const bool multifilter = op.dimensions != kernel.dimensions;
	std::vector<size_t> acc_sizes = calc_acc_sizes(a);
	std::vector<size_t> acc_sizes_pred = calc_acc_sizes(op);
	std::vector<size_t> acc_sizes_kernel = calc_acc_sizes(kernel);
	acc_sizes[op.dimensions - 2] = 1;
	// accumulations of overlapping elements (kernel overlapping itself)
	std::vector<size_t> acc_overlapping(op.dimensions - 1);
	acc_overlapping[acc_overlapping.size() - 1] = 1;
	for (int i = acc_overlapping.size() - 2; i >= 0; i--) {
		acc_overlapping[i] =
			MAX_VAL(1, (long)std::ceil(
						   (double)kernel.shape[multifilter ? i + 2 : i + 1] /
						   (double)steps[i + 1])) *
			acc_overlapping[i + 1];
	}
	// First dimension overlap
	const size_t overlapping =
		MAX_VAL(1, (long)std::ceil((double)kernel.shape[multifilter ? 1 : 0] /
								   (double)steps[0])) *
		acc_overlapping[0];
	for (size_t filter = 0; filter < (multifilter ? kernel.shape[0] : 1);
		 filter++) {
		for (size_t i = from; i < from + size; i++) {
			T res = 0;
			bool in_steps = true;
			bool started_counting = false;
			// get base indices
			size_t keri = 0;
			size_t adji = 0;
			for (int d = 0; d < op.dimensions - 1; d++) {
				const size_t di = (d == 0 ? i : i % acc_sizes_pred[d - 1]) /
								  acc_sizes_pred[d];
				// first kernel element is the offset from di to the first
				// kernel that overlaps it
				const size_t ki = di - (di / steps[d]) * steps[d];
				// if this index is outside the kernel size -> i is not
				// overlapped by a kernel
				if (ki >= kernel.shape[multifilter ? d + 1 : d]) {
					in_steps = false;
					break;
				}
				// first window for this index
				const size_t wdf = (size_t)std::ceil(
					(std::max(0l,
							  (long)di -
								  (long)kernel.shape[multifilter ? d + 1 : d] +
								  1) /
					 (double)steps[d]));
				keri += ki * acc_sizes_kernel[multifilter ? d + 1 : d];
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
						const size_t di =
							(d == 0 ? i : i % acc_sizes_pred[d - 1]) /
							acc_sizes_pred[d];
						const size_t io =
							(d == 0 ? o : o % acc_overlapping[d - 1]) /
							acc_overlapping[d];
						const size_t ao =
							(d == 0 ? actual_overlapping
									: actual_overlapping %
										  acc_overlapping[d - 1]) /
							acc_overlapping[d];
						// check if kernel offset is feasible (the kernel we
						// take the offset to is in bounds)
						const size_t ki =
							(d == 0
								 ? keri
								 : keri %
									   acc_sizes_kernel[multifilter ? d
																	: d - 1]) /
							acc_sizes_kernel[multifilter ? d + 1 : d];
						if (di + kernel.shape[multifilter ? d + 1 : d] -
								(ki + io * steps[d]) >
							op.shape[d]) {
							// those cases are no real windows, only skip them
							// if there haven't been real windows yet
							if (!started_counting) {
								actual_overlapping--;
							}
							skip_kernel = true;
							break;
						} else if (ki + io * steps[d] >=
									   kernel.shape[multifilter ? d + 1 : d] ||
								   di < ki + io * steps[d]) {
							skip_kernel = true;
							break;
						}
						adjo += ao * acc_sizes[d];
						kero += io * steps[d] *
								acc_sizes_kernel[multifilter ? d + 1 : d];
					}
					if (!skip_kernel) {
						started_counting = true;
						res += data1[gnp1->operation.op_type == FGEN_CONSTANT
										 ? 0
										 : filter * acc_sizes_kernel[0] + keri +
											   kero] *
							   data2[gnp2->operation.op_type == FGEN_CONSTANT
										 ? 0
										 : adjo + adji];
					}
					actual_overlapping++;
				}
			}
			if (filter == 0)
				result[i] = res;
			else
				result[i] += res;
		}
	}
}
int GradientConvolve1Impl::generate_ocl_lazy(
	const FGraphNode *node, std::string name,
	OCLLazyCodegenState &compiler_state) {
	FGraphNode *gnp2 = node->predecessors[1];
	FGraphNode *gnp1 = node->predecessors[0];
	string par1 = compiler_state.findOrInsertParameter(gnp1),
		   par2 = compiler_state.findOrInsertParameter(gnp2);
	const FOperation op = node->operation;
	const FOperation kernel = gnp1->operation, a = gnp2->operation;
	const unsigned int *steps = (unsigned int *)op.additional_data;
	const bool multifilter = op.dimensions != kernel.dimensions;
	// calculate accumulated sizes for result (pred), kernel and a
	// (adjacent)
	std::vector<size_t> acc_sizes = calc_acc_sizes(a);
	std::vector<size_t> acc_sizes_pred = calc_acc_sizes(op);
	std::vector<size_t> acc_sizes_kernel = calc_acc_sizes(kernel);
	acc_sizes[op.dimensions - 2] = 1;
	size_t a_num_elems = 1;
	for (long d = a.dimensions - 1; d >= 0; d--)
		a_num_elems *= a.shape[d];
	// accumulations of overlapping elements (kernel overlapping
	// itself)
	std::vector<size_t> acc_overlapping(op.dimensions - 1);
	acc_overlapping[acc_overlapping.size() - 1] = 1;
	for (int i = acc_overlapping.size() - 2; i >= 0; i--) {
		acc_overlapping[i] =
			std::max(1l, (long)std::ceil(
							 (double)kernel.shape[multifilter ? i + 2 : i + 1] /
							 (double)steps[i + 1])) *
			acc_overlapping[i + 1];
	}
	// First dimension overlap
	const size_t overlapping =
		std::max(1l,
				 (long)std::ceil((double)kernel.shape[0] / (double)steps[0])) *
		acc_overlapping[0];
	const string type = type_string(op.data_type);
	Twine convc;
	convc += type + " " + name + " = 0;\nfor(long filter=0;filter<" +
			 to_string(multifilter ? kernel.shape[0] : 1) + ";filter++){";
	convc += "int in_steps = 1, started_counting = 0;\n"
			 "long keri = 0, adji = 0;\n";
	for (int d = 0; d < op.dimensions - 1; d++) {
		convc += "if(in_steps){\nlong di = (";
		if (d == 0)
			convc += "index";
		else
			convc += "index%" + to_string(acc_sizes_pred[d - 1]);
		convc += ") / " + to_string(acc_sizes_pred[d]) +
				 ";\n"
				 "long ki = di - (di / " +
				 to_string(steps[d]) + ")*" + to_string(steps[d]) +
				 ";\n"
				 "if (ki >= " +
				 to_string(kernel.shape[multifilter ? d + 1 : d]) +
				 ") {"
				 " in_steps = 0; }\n"
				 "keri += ki * " +
				 to_string(acc_sizes_kernel[multifilter ? d + 1 : d]) +
				 ";\n"
				 "adji += (long)ceil(max(0l, di - " +
				 to_string(kernel.shape[multifilter ? d + 1 : d] - 1) +
				 ") / (double)" + to_string(steps[d]) + ") * " +
				 to_string(acc_sizes[d]) + ";\n}\n";
	}
	convc += "if(in_steps){\n long actual_overlapping = 0;\n keri "
			 "+= index % " +
			 to_string(op.shape[op.dimensions - 1]) +
			 ";\n for(long o = 0; o < " + to_string(overlapping) +
			 "; o++){\n  int skip_kernel = 0;\n "
			 " long adjo = "
			 "0, kero = 0;\n";
	for (int d = 0; d < op.dimensions - 1; d++) {
		convc += "  if(!skip_kernel){\n   const long di = (";
		if (d == 0)
			convc += "index";
		else
			convc += "index%" + to_string(acc_sizes_pred[d - 1]);
		convc += ")/" + to_string(acc_sizes_pred[d]) +
				 ";\n"
				 "   const long io = (";
		if (d == 0)
			convc += "o";
		else
			convc += "o%" + to_string(acc_overlapping[d - 1]);
		convc += ")/" + to_string(acc_overlapping[d]) +
				 ";\n"
				 "   const long ao = (";
		if (d == 0)
			convc += "actual_overlapping";
		else
			convc += "actual_overlapping%" + to_string(acc_overlapping[d - 1]);
		convc += ")/" + to_string(acc_overlapping[d]) +
				 ";\n"
				 "   const long ki = (";
		if (d == 0)
			convc += "keri";
		else
			convc +=
				"keri%" + to_string(acc_sizes_kernel[multifilter ? d : d - 1]);
		convc +=
			")/" + to_string(acc_sizes_kernel[multifilter ? d + 1 : d]) +
			";\n"
			"   if(di + " +
			to_string(kernel.shape[multifilter ? d + 1 : d]) +
			" - (ki + io * " + to_string(steps[d]) + ") > " +
			to_string(op.shape[d]) +
			"){\n"
			"    if(!started_counting) actual_overlapping--;\n"
			"    skip_kernel = true;\n"
			"   }else if(ki + io * " +
			to_string(steps[d]) +
			" >= " + to_string(kernel.shape[multifilter ? d + 1 : d]) +
			" || di < ki + io * " + to_string(steps[d]) +
			"){\n"
			"    skip_kernel = true;\n"
			"   }\n"
			"   adjo += ao * " +
			to_string(acc_sizes[d]) +
			";\n"
			"   kero += io * " +
			to_string(steps[d] * acc_sizes_kernel[multifilter ? d + 1 : d]) +
			";\n  }\n";
	}
	string ind1 =
		"filter * " + to_string(acc_sizes_kernel[0]) + " + keri + kero";
	string ind2 = "adji + adjo";
	if (gnp1->operation.op_type == FGEN_CONSTANT)
		ind1 = "0";
	if (gnp2->operation.op_type == FGEN_CONSTANT)
		ind2 = "0";
	convc += "  if(!skip_kernel){\n"
			 "   started_counting = true;\n"
			 "   " +
			 name + " += " + par1 + "[" + ind1 + "] * " + par2 + "[" + ind2 +
			 "];\n"
			 " }\n"
			 " actual_overlapping++;\n}\n}\n}\n";
	compiler_state.code.prepend(convc);
	return OCL_LAZY_DONT_PUSH_PREDS;
}
std::string GradientConvolve1Impl::generate_ocl_parameters_eager(
	FType res_type, std::vector<FType> parameter_types) {
	return ", const __global " + type_string(parameter_types[0]) +
		   "* P0"
		   ", const long num_entries0, const int dimensions0, const "
		   "__global " +
		   type_string(parameter_types[1]) +
		   "* P1, const long num_entries1, const int dimensions1, const int "
		   "dimensionsR"
		   ", __constant long* acc_sizes_pred, "
		   "__constant long* acc_sizes_kernel"
		   ", __constant long* acc_sizes, __constant long* acc_overlapping"
		   ", __constant int* steps, __constant long* op_shape, __constant "
		   "long* kernel_shape";
}
std::string
GradientConvolve1Impl::generate_ocl_eager(FType res_type,
										  std::vector<FType> parameter_types) {
	return "if(index >= num_entriesR) return;\n"
		   "const bool multifilter = dimensionsR != dimensions0;\n"
		   "const long overlapping = max(1l, "
		   "(long)ceil(kernel_shape[multifilter ? 1 : 0] / "
		   "(double)steps[0])) * acc_overlapping[0];\n" +
		   type_string(res_type) +
		   " res = 0;\n"
		   "int in_steps = true;\n"
		   "long keri = 0;\n"
		   "long adji = 0;\n"
		   "for(int d = 0; d < dimensionsR-1; d++){\n"
		   " const long di = (d == 0 ? index : index % acc_sizes_pred[d-1]) / "
		   "acc_sizes_pred[d];\n"
		   " const long ki = di - (di / steps[d]) * steps[d];\n"
		   " if(ki >= kernel_shape[multifilter ? d + 1 : d]){\n"
		   "  in_steps = false;\n"
		   "  break;\n"
		   " }\n"
		   " const long wdf = (long)ceil(max(0l, di - kernel_shape[multifilter "
		   "? d + 1 : d] + 1) / "
		   "(double)steps[d]);\n"
		   " keri += ki * acc_sizes_kernel[multifilter ? d + 1 : d];\n"
		   " adji += wdf * acc_sizes[d];\n"
		   "}\n"
		   "if(in_steps){\n"
		   " keri += index % op_shape[dimensionsR - 1];\n"
		   " for(long filter = 0; filter < (multifilter ? kernel_shape[0] : "
		   "1); filter++){\n"
		   "  int started_counting = false;\n"
		   "  long actual_overlapping = 0;\n"
		   "  for(long o = 0; o < overlapping; o++){\n"
		   "   long adjo = 0;\n"
		   "   long kero = 0;\n"
		   "   int skip_kernel = false;\n"
		   "   for(int d = 0; d < dimensionsR - 1; d++){\n"
		   "    const long di = (d == 0 ? index : index % acc_sizes_pred[d-1]) "
		   "/ acc_sizes_pred[d];\n"
		   "    const long io = (d == 0 ? o : o % acc_overlapping[d-1]) / "
		   "acc_overlapping[d];\n"
		   "    const long ao = (d == 0 ? actual_overlapping : "
		   "actual_overlapping % acc_overlapping[d-1]) / acc_overlapping[d];\n"
		   "    const long ki = (d == 0 ? keri : keri % "
		   "acc_sizes_kernel[multifilter ? d : d-1]) "
		   "/ acc_sizes_kernel[multifilter ? d + 1 : d];\n"
		   "    if(di+kernel_shape[multifilter ? d + 1 : d]-(ki+io*steps[d]) > "
		   "op_shape[d]){\n"
		   "     if(!started_counting) actual_overlapping--;\n"
		   "     skip_kernel = true;\n"
		   "     break;\n"
		   "    }else if(ki+io*steps[d] >= kernel_shape[multifilter ? d + 1 : "
		   "d] || di < "
		   "ki+io*steps[d]){\n"
		   "     skip_kernel = true;\n"
		   "     break;\n"
		   "    }\n"
		   "    adjo += ao*acc_sizes[d];\n"
		   "    kero += io*steps[d]*acc_sizes_kernel[multifilter ? d + 1 : "
		   "d];\n"
		   "   }\n"
		   "   if(!skip_kernel){\n"
		   "    started_counting = true;\n"
		   "    "
		   "res+=P0[(filter*acc_sizes_kernel[0]+kero+keri)%num_entries0"
		   "]*P1[(adjo+adji)%num_entries1];\n"
		   "   }\n"
		   "   actual_overlapping++;\n"
		   "  }\n"
		   " }\n"
		   "}\n"
		   "R[index] = res;\n";
}
void GradientConvolve1Impl::push_additional_kernel_parameters(
	FGraphNode *node, cl_kernel kernel, cl_context context, int &par_index,
	std::list<cl_mem> &to_free) {
	const FOperation op = node->operation;
	const FGraphNode *gnp1 = node->predecessors[0],
					 *gnp2 = node->predecessors[1];
	const FOperation kernel_op = gnp1->operation, a = gnp2->operation;
	unsigned int *steps = (unsigned int *)op.additional_data;
	// push dimensionsR
	if (clSetKernelArg(kernel, par_index++, sizeof(int),
					   (void *)&op.dimensions) != CL_SUCCESS) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Could not load Argument to kernel!");
		return;
	}
	to_free.push_back(calc_and_push_acc_size(op.dimensions, op.shape, kernel,
											 context, par_index));
	to_free.push_back(calc_and_push_acc_size(
		kernel_op.dimensions, kernel_op.shape, kernel, context, par_index));
	to_free.push_back(calc_and_push_acc_size(a.dimensions, a.shape, kernel,
											 context, par_index));

	const bool multifilter = op.dimensions != kernel_op.dimensions;
	std::vector<size_t> acc_overlapping(op.dimensions - 1);
	acc_overlapping[acc_overlapping.size() - 1] = 1;
	for (int i = acc_overlapping.size() - 2; i >= 0; i--) {
		acc_overlapping[i] =
			std::max(1l,
					 (long)std::ceil(
						 (double)kernel_op.shape[multifilter ? i + 2 : i + 1] /
						 (double)steps[i + 1])) *
			acc_overlapping[i + 1];
	}
	to_free.push_back(push_array(acc_overlapping.size(), acc_overlapping.data(),
								 kernel, context, par_index));
	to_free.push_back(
		push_array(op.dimensions - 1, steps, kernel, context, par_index));
	to_free.push_back(
		push_array(op.dimensions, op.shape, kernel, context, par_index));
	to_free.push_back(push_array(kernel_op.dimensions, kernel_op.shape, kernel,
								 context, par_index));
}
FGraphNode *GradientConvolve2Impl::local_gradient(FGraphNode *y, int dx_i,
												  FGraphNode *prev_adj) {
	FGraphNode *a = y->predecessors[0];
	FGraphNode *b = y->predecessors[1];
	const unsigned int *steps = (unsigned int *)y->operation.additional_data;
	const bool multifilter = a->operation.dimensions != b->operation.dimensions;
	if (0 == dx_i) {
		return ConvolveImpl::gradient_convolve1(a, b, prev_adj, steps);
	} else if (1 == dx_i) {
		FGraphNode *sliding_window =
			fmul(fsliding_window(a, y->operation.shape, steps), prev_adj);
		// now reduce each window
		for (int d = sliding_window->operation.dimensions; d > 0; d--) {
			sliding_window = freduce_sum(sliding_window, d);
		}
		return freshape(sliding_window, b->operation.shape,
						b->operation.dimensions);
	} else {
		return nullptr;
	}
}
/**
 * Calculates total number of elemens (can be retreived by providing a pointer
 * to `total_elems`) and deceides if an by how many threads a element should be
 * split.
 */
static int
size_multiplier_convolve_kernel_gradient(const FGraphNode *node,
										 size_t *total_elems = nullptr) {
	const FOperation op = node->operation;
	const FGraphNode *gnp1 = node->predecessors[0],
					 *gnp2 = node->predecessors[1];
	const FOperation pred = gnp1->operation, prev_adj = gnp2->operation;
	const bool multifilter = op.dimensions > pred.dimensions;
	std::vector<size_t> acc_sizes_windows(multifilter ? prev_adj.dimensions - 1
													  : prev_adj.dimensions);
	acc_sizes_windows[acc_sizes_windows.size() - 1] = 1;
	for (int i = acc_sizes_windows.size() - 2; i >= 0; i--)
		acc_sizes_windows[i] = acc_sizes_windows[i + 1] * prev_adj.shape[i + 1];
	// total number of windows
	const size_t windows = acc_sizes_windows[0] * prev_adj.shape[0];
	// total number of elements
	size_t num_elems = 1;
	for (int d = 0; d < node->operation.dimensions; d++)
		num_elems *= node->operation.shape[d];
	if (total_elems)
		*total_elems = num_elems;
	// calculate multiplicator
	if (num_elems <= 500 && windows >= 16)
		return 4;
	else if (num_elems < 2000 && windows >= 8)
		return 2;
	else
		return 1;
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

	// multiplication coefficient
	const int c = size_multiplier_convolve_kernel_gradient(curr, nullptr);
	const FOperation op = curr->operation;
	const FGraphNode *gnp1 = curr->predecessors[0],
					 *gnp2 = curr->predecessors[1];
	const FOperation pred = gnp1->operation, prev_adj = gnp2->operation;
	const std::vector<size_t> acc_sizes_pred = calc_acc_sizes(pred);
	const std::vector<size_t> acc_sizes_kernel = calc_acc_sizes(op);
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
	const size_t window_work_load = windows / c;
	for (size_t i_m = from; i_m < from + size; i_m++) {
		const int i = i_m / c;
		const int window_thread = i_m % c;
		const size_t to = window_thread == (c - 1)
							  ? windows
							  : (window_thread + 1) * window_work_load;
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
		auto target = atomic_ref<T>(result[i]);
		// iterate over windows = adjoint elements in first dimensions
		// we split windows for the thread iterations
		for (size_t w = window_thread * window_work_load; w < to; w++) {
			// calculate start value of window for pred
			size_t a = 0;
			for (int j = 0; j < acc_sizes_windows.size(); j++) {
				size_t wj = (w / acc_sizes_windows[j]) % prev_adj.shape[j];
				a += wj * acc_sizes_pred[j] * steps[j];
			}
			const T res =
				data1[gnp1->operation.op_type == FGEN_CONSTANT ? 0
															   : a + a_offset] *
				data2[gnp2->operation.op_type == FGEN_CONSTANT
						  ? 0
						  : w * num_filter + f];
			target += res;
		}
	}
}
size_t GradientConvolve2Impl::deploy_as_many_elements(const FGraphNode *node) {
	size_t num_elems;
	// calculate multiplicator
	const int c = size_multiplier_convolve_kernel_gradient(node, &num_elems);
	return c * num_elems;
}
int GradientConvolve2Impl::generate_ocl_lazy(
	const FGraphNode *node, std::string name,
	OCLLazyCodegenState &compiler_state) {
	const FOperation op = node->operation;
	FGraphNode *gnp1 = node->predecessors[0], *gnp2 = node->predecessors[1];
	int vari = compiler_state.variable_index;
	const string par1 = "v" + to_string(++compiler_state.variable_index);
	const string par2 = compiler_state.findOrInsertParameter(gnp2);
	const FOperation pred = gnp1->operation, prev_adj = gnp2->operation;
	const vector<size_t> acc_sizes_pred = calc_acc_sizes(pred);
	const vector<size_t> acc_sizes_kernel = calc_acc_sizes(op);
	const bool multifilter = op.dimensions > pred.dimensions;
	const string type = type_string(op.data_type);
	const unsigned int num_filter = multifilter ? op.shape[0] : 1;
	// like accumulated sizes for prev_adj but without filter in
	// multifilter context
	vector<size_t> acc_sizes_windows(multifilter ? prev_adj.dimensions - 1
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
	const string a_offset = "a_offset" + to_string(vari);
	const string w = "w" + to_string(vari);
	const string a = "a" + to_string(vari);
	std::string grad_code =
		type + " " + name + " = 0;\nlong " + a_offset + " = 0";

	for (int j = multifilter ? 1 : 0; j < op.dimensions; j++) {
		grad_code += "+((index/" + to_string(acc_sizes_kernel[j]) + ")%" +
					 to_string(op.shape[j]) + ")*" +
					 to_string(acc_sizes_pred[multifilter ? j - 1 : j]);
	}
	grad_code += ";\n"
				 "for(long " +
				 w + " = 0; " + w + " < " + to_string(windows) + "; " + w +
				 "++){\n"
				 " long " +
				 a + " = 0";
	for (int j = 0; j < acc_sizes_windows.size(); j++) {
		grad_code += "+((" + w + "/" + to_string(acc_sizes_windows[j]) + ")%" +
					 to_string(prev_adj.shape[j]) + ")*" +
					 to_string(acc_sizes_pred[j] * steps[j]);
	}
	grad_code += ";\n";
	const string old_idx = "old_idx" + to_string(compiler_state.num_indices++);
	grad_code += " long " + old_idx +
				 " = index;\n"
				 " index = " +
				 a + " + " + a_offset + ";\n";
	const string f =
		multifilter ? old_idx + " / " + to_string(num_elems_kernel) : "0";
	compiler_state.todo.push_front({nullptr, grad_code});
	compiler_state.todo.push_front({gnp1, par1});
	string ind2 = w + " * " + to_string(num_filter) + " + " + f;
	if (gnp2->operation.op_type == FGEN_CONSTANT)
		ind2 = "0";
	compiler_state.code.prepend(" " + name + "+=" + par1 + "*" + par2 + "[" +
								ind2 +
								"];\n"
								" index = " +
								old_idx + ";\n}\n");
	return OCL_LAZY_DONT_PUSH_PREDS;
}
std::string GradientConvolve2Impl::generate_ocl_parameters_eager(
	FType res_type, std::vector<FType> parameter_types) {
	return ", const __global " + type_string(parameter_types[0]) +
		   "* P1"
		   ", const long num_entries1, const int dimensions1, const __global " +
		   type_string(parameter_types[1]) +
		   "* P2, const long num_entries2, const int dimensions2, "
		   "const int dimensions0, "
		   "__constant long* acc_sizes_pred, __constant long* "
		   "acc_sizes_kernel, "
		   "__constant long* acc_sizes_windows, __constant int* steps, "
		   "__constant long* op_shape, __constant long* prev_adj_shape, const "
		   "int c";
}
std::string
GradientConvolve2Impl::generate_ocl_eager(FType res_type,
										  std::vector<FType> parameter_types) {
	return "const long i_m = index;\nindex /= c;\n"
		   "if(index >= num_entriesR) return;\n"
		   "const int window_thread = i_m % c;\n"
		   "const bool multifilter = dimensions0 > dimensions1;\n"
		   "const long windows = acc_sizes_windows[0] * prev_adj_shape[0];\n"
		   "const long window_work_load = windows / c;\n"
		   "const long to = window_thread == (c-1) ? windows : (window_thread "
		   "+ 1) * window_work_load;\n"
		   "const long num_elems_kernel = multifilter ? acc_sizes_kernel[0] : "
		   "acc_sizes_kernel[0] * op_shape[0];\n"
		   "const int num_filter = multifilter ? op_shape[0] : 1;\n"
		   "const long f = multifilter ? index / num_elems_kernel : 0;\n"
		   "long a_offset = 0;\n"
		   "for(int j = multifilter ? 1 : 0; j < dimensions0; j++){\n"
		   " const long ki = (index / acc_sizes_kernel[j]) % op_shape[j];\n"
		   " a_offset += ki * acc_sizes_pred[multifilter ? j - 1 : j];\n"
		   "}\n"
		   //"R[index] = 0;\n"
		   + type_string(res_type) +
		   " res = 0;\n"
		   "for(long w = window_thread * window_work_load; w < to; w++){\n"
		   " long a = 0;"
		   " for(int j = 0; j < (multifilter ? dimensions2 - 1 : "
		   "dimensions2); "
		   "j++){\n"
		   "  const long wj = (w / acc_sizes_windows[j]) % "
		   "prev_adj_shape[j];\n"
		   "  a += wj * acc_sizes_pred[j] * steps[j];\n"
		   " }\n"
		   " res += P1[(a + a_offset) % num_entries1] * P2[(w * "
		   "num_filter + f) % num_entries2];\n"
		   "}\n"
		   "for(int t = 0; t < c; t++){\n"
		   " barrier(CLK_GLOBAL_MEM_FENCE);\n"
		   " if(window_thread == t)\n"
		   "  R[index] += res;\n"
		   "}\n";
}
void GradientConvolve2Impl::push_additional_kernel_parameters(
	FGraphNode *node, cl_kernel kernel, cl_context context, int &par_index,
	std::list<cl_mem> &to_free) {
	const FOperation op = node->operation;
	const FGraphNode *gnp1 = node->predecessors[0],
					 *gnp2 = node->predecessors[1];
	const FOperation pred = gnp1->operation, prev_adj = gnp2->operation;
	cl_int err_code;
	// dimensions0
	if (clSetKernelArg(kernel, par_index++, sizeof(int),
					   (void *)&node->operation.dimensions) != CL_SUCCESS) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Could not load Argument to kernel!");
		return;
	}
	const bool multifilter = op.dimensions > pred.dimensions;
	to_free.push_back(calc_and_push_acc_size(pred.dimensions, pred.shape,
											 kernel, context, par_index));
	to_free.push_back(calc_and_push_acc_size(op.dimensions, op.shape, kernel,
											 context, par_index));
	to_free.push_back(calc_and_push_acc_size(
		multifilter ? prev_adj.dimensions - 1 : prev_adj.dimensions,
		prev_adj.shape, kernel, context, par_index));
	unsigned int *steps = (unsigned int *)op.additional_data;
	// allocate steps
	cl_mem steps_mem =
		clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					   (pred.dimensions - 1) * sizeof(int), steps, &err_code);
	if (!steps_mem)
		flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
							  std::to_string(err_code));
	// allocate shape0
	cl_mem op_shape_mem =
		clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					   op.dimensions * sizeof(long), op.shape, &err_code);
	if (!op_shape_mem)
		flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
							  std::to_string(err_code));
	// allocate prev_adj_shape
	cl_mem prev_adj_shape = clCreateBuffer(
		context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		prev_adj.dimensions * sizeof(long), prev_adj.shape, &err_code);
	if (!prev_adj_shape)
		flogging(F_ERROR, "Could not load Argument to kernel! Error Code: " +
							  std::to_string(err_code));
	for (cl_mem mem : {steps_mem, op_shape_mem, prev_adj_shape}) {
		to_free.push_back(mem);
		if (clSetKernelArg(kernel, par_index++, sizeof(cl_mem), (void *)&mem) !=
			CL_SUCCESS)
			flogging(F_ERROR,
					 "Could not load Argument to kernel! Error Code: " +
						 std::to_string(err_code));
	}
	const int c = size_multiplier_convolve_kernel_gradient(node, nullptr);
	if (clSetKernelArg(kernel, par_index++, sizeof(int), &c) != CL_SUCCESS) {
		setErrorType(OCL_ERROR);
		flogging(F_ERROR, "Could not load Argument to kernel!");
		return;
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
