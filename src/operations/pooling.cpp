#include "pooling.hpp"

#define MIN_VAL(x, y) (x < y ? x : y)
#define MAX_VAL(x, y) (x < y ? y : x)

using namespace std;

template <typename T>
static void pooling(T *__restrict__ result, const T *__restrict__ data,
					size_t from, size_t size, const FGraphNode *curr) {
	const FOperation op = curr->operation;
	const FGraphNode *gnp1 = curr->predecessors[0];
	const FOperation pred = gnp1->operation;
	const FSlidingWindow *window = (FSlidingWindow *)op.additional_data;
	// calculate accumulated sizes for result, kernel and source (pred)
	const std::vector<size_t> acc_sizes = calcAccSizes(op);
	const std::vector<size_t> acc_sizes_pred = calcAccSizes(pred);
	size_t kernel_num_elems = window->size[op.dimensions - 1];
	std::vector<size_t> acc_sizes_kernel = std::vector<size_t>(op.dimensions);
	acc_sizes_kernel[op.dimensions - 1] = 1;
	for (int d = op.dimensions - 2; d >= 0; d--) {
		acc_sizes_kernel[d] = acc_sizes_kernel[d + 1] * window->size[d + 1];
		kernel_num_elems *= window->size[d];
	}
	for (size_t i = from; i < from + size; i++) {
		// base index for source
		size_t j = 0;
		for (unsigned int d = 0; d < op.dimensions; d++) {
			// get dimension index
			const size_t di =
				(d == 0 ? i : i % acc_sizes[d - 1]) / acc_sizes[d];
			// reproject
			j += di * window->step[d] * acc_sizes_pred[d];
		}
		T res = 0;
		if (op.op_type == FPOOLING_MAX) {
			res = std::numeric_limits<T>::lowest();
		}
		for (size_t k = 0; k < kernel_num_elems; k++) {
			size_t o = 0; // source offset
			for (unsigned int d = 0; d < op.dimensions; d++) {
				const size_t dk = (d == 0 ? k : k % acc_sizes_kernel[d - 1]) /
								  acc_sizes_kernel[d];
				o += dk * acc_sizes_pred[d];
			}
			// full reduction in last dimension so iterate over it too
			for (size_t ld = 0; ld < pred.shape[pred.dimensions - 1]; ld++) {
				switch (op.op_type) {
				case FPOOLING_SUM:
					res += data[j + o + ld];
					break;
				case FPOOLING_MAX:
					res = MAX_VAL(data[j + o + ld], res);
				default:
					break;
				}
			}
		}
		result[i] = res;
	}
}
static int pooling_gpu(const FGraphNode *node, std::string name,
					   OCLLazyCodegenState &compiler_state) {
	const FOperation op = node->operation;
	const FGraphNode *gnp1 = node->predecessors[0];
	const FOperation pred = gnp1->operation;
	const FSlidingWindow *window = (FSlidingWindow *)op.additional_data;
	const string type = typeString(node->operation.data_type);
	// calculate accumulated sizes for result, kernel and source
	// (pred)
	const vector<size_t> acc_sizes = calcAccSizes(op);
	const vector<size_t> acc_sizes_pred = calcAccSizes(pred);
	size_t kernel_num_elems = window->size[op.dimensions - 1];
	vector<size_t> acc_sizes_kernel = vector<size_t>(op.dimensions);
	acc_sizes_kernel[op.dimensions - 1] = 1;
	for (int d = op.dimensions - 2; d >= 0; d--) {
		acc_sizes_kernel[d] = acc_sizes_kernel[d + 1] * window->size[d + 1];
		kernel_num_elems *= window->size[d];
	}
	const string base_ind =
		"base_ind" + to_string(compiler_state.variable_index);
	Twine pooling_code =
		type + " " + name + " = " +
		(op.op_type == FPOOLING_SUM ? "0" : minForType(op.data_type)) +
		";\nlong " + base_ind + " = 0";
	for (int d = 0; d < op.dimensions; d++) {
		pooling_code +=
			"+" +
			(d == 0 ? "index" : "(index%" + to_string(acc_sizes[d - 1]) + ")") +
			"/" + to_string(acc_sizes[d]) + " * " +
			to_string(window->step[d] * acc_sizes_pred[d]);
	}
	const std::string k = "k" + to_string(compiler_state.variable_index);
	const std::string o = "o" + to_string(compiler_state.variable_index);
	pooling_code += ";\n"
					"for(long " +
					k + " = 0; " + k + " < " + to_string(kernel_num_elems) +
					"; " + k +
					"++){\n"
					" long " +
					o + " = 0";
	for (int d = 0; d < op.dimensions; d++) {
		pooling_code +=
			"+" +
			(d == 0
				 ? k
				 : "(" + k + "%" + to_string(acc_sizes_kernel[d - 1]) + ")") +
			"/" + to_string(acc_sizes_kernel[d]) + "*" +
			to_string(acc_sizes_pred[d]);
	}
	const std::string ld = "ld" + to_string(compiler_state.variable_index);
	const unsigned int old_idx = compiler_state.num_indices++;
	pooling_code += ";\n for(long " + ld + " = 0; " + ld + " < " +
					to_string(pred.shape[pred.dimensions - 1]) + "; " + ld +
					"++){\n"
					"  long old_idx" +
					to_string(old_idx) +
					" = index;\n"
					"  index = " +
					base_ind + "+" + o + "+" + ld + ";\n";
	compiler_state.index_defs += pooling_code;
	compiler_state.code.prepend(
		"  index = old_idx" + to_string(old_idx) + ";\n  " + name +
		(op.op_type == FPOOLING_SUM
			 ? " += v" + to_string(compiler_state.variable_index + 1)
			 : " = max(" + name + ", v" +
				   to_string(compiler_state.variable_index + 1) + ")") +
		";\n }\n}");
	return 0;
}
template <typename T>
void PoolingSumImpl::unary_expression(T *__restrict__ result,
									  const T *__restrict__ data, size_t from,
									  size_t size, const FGraphNode *curr) {
	pooling(result, data, from, size, curr);
}
int PoolingSumImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
									  OCLLazyCodegenState &compiler_state) {
	return pooling_gpu(node, name, compiler_state);
}
void PoolingSumImpl::execute_cpu(const FGraphNode *node,
								 std::vector<CPUResultData> predecessor_data,
								 void *__restrict__ result, size_t from,
								 size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}

template <typename T>
void PoolingMaxImpl::unary_expression(T *__restrict__ result,
									  const T *__restrict__ data, size_t from,
									  size_t size, const FGraphNode *curr) {
	pooling(result, data, from, size, curr);
}
int PoolingMaxImpl::generate_ocl_lazy(const FGraphNode *node, std::string name,
									  OCLLazyCodegenState &compiler_state) {
	return pooling_gpu(node, name, compiler_state);
}
void PoolingMaxImpl::execute_cpu(const FGraphNode *node,
								 std::vector<CPUResultData> predecessor_data,
								 void *__restrict__ result, size_t from,
								 size_t size) {
	UNARY_EXECUTE_MONOTON_IMPL
}
template <typename T>
void GradientPoolingMax::execute_cpu_typed(
	const FGraphNode *node, std::vector<CPUResultData> predecessor_data,
	T *__restrict__ result, size_t from, size_t size) {
	const FOperation op = node->operation;
	const FGraphNode *gnp1 = node->predecessors[0],
					 *gnp2 = node->predecessors[1],
					 *gnp3 = node->predecessors[2];
	const FSlidingWindow *window =
		(FSlidingWindow *)gnp1->operation.additional_data;
	const FOperation a = gnp2->operation;
	const FOperation image = gnp3->operation;
	const void *data1 = predecessor_data[0].data; // pooling
	const void *data2 = predecessor_data[1].data; // adjoint
	const void *data3 = predecessor_data[2].data; // image
	const unsigned int *steps = window->step;
	// calculate accumulated sizes for result (pred), kernel and a
	// (adjacent)
	std::vector<size_t> acc_sizes = calcAccSizes(a);
	std::vector<size_t> acc_sizes_pred = calcAccSizes(op);
	acc_sizes[op.dimensions - 2] = 1;
	std::vector<size_t> acc_sizes_kernel(op.dimensions);
	acc_sizes_kernel[acc_sizes_kernel.size() - 1] = 1;
	acc_sizes_kernel[acc_sizes_kernel.size() - 2] = op.shape[op.dimensions - 1];
	for (int i = acc_sizes_kernel.size() - 3; i >= 0; i--)
		acc_sizes_kernel[i] = window->size[i + 1] * acc_sizes_kernel[i + 1];
	// accumulations of overlapping elements (kernel overlapping itself)
	std::vector<size_t> acc_overlapping(op.dimensions - 1);
	acc_overlapping[acc_overlapping.size() - 1] = 1;
	for (int i = acc_overlapping.size() - 2; i >= 0; i--) {
		acc_overlapping[i] =
			MAX_VAL(1, (long)std::ceil((double)window->size[i + 1] /
									   (double)steps[i + 1])) *
			acc_overlapping[i + 1];
	}
	// First dimension overlap
	const size_t overlapping =
		MAX_VAL(1,
				(long)std::ceil((double)window->size[0] / (double)steps[0])) *
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
			if (ki >= window->size[d]) {
				in_steps = false;
				break;
			}
			// first window for this index
			const size_t wdf = (size_t)std::ceil(
				(std::max(0l, (long)di - (long)window->size[d] + 1) /
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
					if (di + window->size[d] - (ki + io * steps[d]) >
						op.shape[d]) {
						// those cases are no real windows, only skip them
						// if there haven't been real windows yet
						if (!started_counting) {
							actual_overlapping--;
						}
						skip_kernel = true;
						break;
					} else if (ki + io * steps[d] >= window->size[d] ||
							   di < ki + io * steps[d]) {
						skip_kernel = true;
						break;
					}
					adjo += ao * acc_sizes[d];
				}
				// if value in image and corresponding pooling are equal
				bool equal;
				switch (predecessor_data[2].type) {
				case F_INT32:
					equal = ((const int *__restrict__)data3)[i] ==
							((const int *__restrict__)data1)[adjo + adji];
					break;
				case F_INT64:
					equal = ((const long *__restrict__)data3)[i] ==
							((const long *__restrict__)data1)[adjo + adji];
					break;
				case F_FLOAT32:
					equal = ((const float *__restrict__)data3)[i] ==
							((const float *__restrict__)data1)[adjo + adji];
					break;
				case F_FLOAT64:
					equal = ((const double *__restrict__)data3)[i] ==
							((const double *__restrict__)data1)[adjo + adji];
					break;
				}
				if (!skip_kernel && equal) {
					started_counting = true;
					res += ((const T *__restrict__)data2)[adjo + adji];
				}
				actual_overlapping++;
			}
		}
		result[i] = res;
	}
}
int GradientPoolingMax::generate_ocl_lazy(const FGraphNode *node,
										  std::string name,
										  OCLLazyCodegenState &compiler_state) {
	FGraphNode *gnp3 = node->predecessors[2];
	FGraphNode *gnp2 = node->predecessors[1];
	FGraphNode *gnp1 = node->predecessors[0];
	const string par1 = compiler_state.findOrInsertParameter(gnp1),
				 par2 = compiler_state.findOrInsertParameter(gnp2),
				 par3 = compiler_state.findOrInsertParameter(gnp3);
	const FOperation op = node->operation;
	const FSlidingWindow *window =
		(FSlidingWindow *)gnp1->operation.additional_data;
	const FOperation a = gnp2->operation;
	const FOperation image = gnp3->operation;
	const unsigned int *steps = window->step;
	const string type = typeString(op.data_type);
	// calculate accumulated sizes for result (pred), kernel and a
	// (adjacent)
	std::vector<size_t> acc_sizes = calcAccSizes(a);
	std::vector<size_t> acc_sizes_pred = calcAccSizes(op);
	acc_sizes[op.dimensions - 2] = 1;
	std::vector<size_t> acc_sizes_kernel(op.dimensions);
	acc_sizes_kernel[acc_sizes_kernel.size() - 1] = 1;
	acc_sizes_kernel[acc_sizes_kernel.size() - 2] = op.shape[op.dimensions - 1];
	for (int i = acc_sizes_kernel.size() - 3; i >= 0; i--)
		acc_sizes_kernel[i] = window->size[i + 1] * acc_sizes_kernel[i + 1];
	// accumulations of overlapping elements (kernel overlapping
	// itself)
	std::vector<size_t> acc_overlapping(op.dimensions - 1);
	acc_overlapping[acc_overlapping.size() - 1] = 1;
	for (int i = acc_overlapping.size() - 2; i >= 0; i--) {
		acc_overlapping[i] =
			std::max(1l, (long)std::ceil((double)window->size[i + 1] /
										 (double)steps[i + 1])) *
			acc_overlapping[i + 1];
	}
	// First dimension overlap
	const size_t overlapping =
		std::max(1l,
				 (long)std::ceil((double)window->size[0] / (double)steps[0])) *
		acc_overlapping[0];
	Twine convc = type + " " + name + " = 0;\n{";
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
				 to_string(window->size[d]) +
				 ") {"
				 " in_steps = 0; }\n"
				 "keri += ki * " +
				 to_string(acc_sizes_kernel[d]) +
				 ";\n"
				 "adji += (long)ceil(max(0l, di - " +
				 to_string(window->size[d] - 1) + ") / (double)" +
				 to_string(steps[d]) + ") * " + to_string(acc_sizes[d]) +
				 ";\n}\n";
	}
	convc += "if(in_steps){\n long actual_overlapping = 0;\n keri "
			 "+= index % " +
			 to_string(op.shape[op.dimensions - 1]) +
			 ";\n for(long o = 0; o < " + to_string(overlapping) +
			 "; o++){\n  int skip_kernel = 0;\n "
			 " long adjo = "
			 "0;\n";
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
			convc += "keri%" + to_string(acc_sizes_kernel[d - 1]);
		convc += ")/" + to_string(acc_sizes_kernel[d]) +
				 ";\n"
				 "   if(di + " +
				 to_string(window->size[d]) + " - (ki + io * " +
				 to_string(steps[d]) + ") > " + to_string(op.shape[d]) +
				 "){\n"
				 "    if(!started_counting) actual_overlapping--;\n"
				 "    skip_kernel = true;\n"
				 "   }else if(ki + io * " +
				 to_string(steps[d]) + " >= " + to_string(window->size[d]) +
				 " || di < ki + io * " + to_string(steps[d]) +
				 "){\n"
				 "    skip_kernel = true;\n"
				 "   }\n"
				 "   adjo += ao * " +
				 to_string(acc_sizes[d]) + ";\n  }\n";
	}
	convc += "  const int equal = " + par3 + "[index] == " + par1 +
			 "[adjo + adji];\n"
			 "  if(!skip_kernel && equal){\n"
			 "   started_counting = true;\n"
			 "   " +
			 name + " += " + par2 +
			 "[adji + adjo];\n"
			 " }\n"
			 " actual_overlapping++;\n}\n}\n}\n";
	compiler_state.code.prepend(convc);
	return OCL_LAZY_DONT_PUSH_PREDS;
}
void GradientPoolingMax::execute_cpu(
	const FGraphNode *node, std::vector<CPUResultData> predecessor_data,
	void *__restrict__ result, size_t from, size_t size) {
	EXECUTE_TYPED_IMPL
}
