#include "implementation.hpp"
#include "simple_arithmetic.hpp"

struct NopImpl : OperationImplementation {
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override {}
		void generate_gpu_lazy() override {}
		void generate_gpu_eager() override {}
};

std::vector<OperationImplementation *>
	OperationImplementation::implementations = {new NopImpl(), // store
												new NopImpl(), // gen random
												new NopImpl(), // gen const
												new NopImpl(), // gen arange
												new AddImpl(), new SubImpl(),
												new MulImpl(), new DivImpl(),
												new PowImpl()};
