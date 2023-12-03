#include "shape_modification.hpp"
#include "src/backend_cpu/cpu_common.hpp"
#include <cstring>

void FlattenImpl::execute_cpu(const FGraphNode *node,
							  std::vector<CPUResultData> predecessor_data,
							  void *__restrict__ result, size_t from,
							  size_t size) {
	const CPUResultData pred = predecessor_data[0];
	switch (predecessor_data[0].type) {
	case F_INT32:
	case F_FLOAT32:
		std::memcpy((int *)result + from, (int *)pred.data + from, 4 * size);
		break;
	case F_INT64:
	case F_FLOAT64:
		std::memcpy((long *)result + from, (long*)pred.data + from, 4 * size);
		break;
	}
}
