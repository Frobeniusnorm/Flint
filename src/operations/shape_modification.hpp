#ifndef FLINT_SHAPE_MODIFICATION_HPP
#define FLINT_SHAPE_MODIFICATION_HPP
#include "implementation.hpp"

struct FlattenImpl : OperationImplementation {
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		void generate_gpu_lazy() override {}
		void generate_gpu_eager() override {}
};

#endif
