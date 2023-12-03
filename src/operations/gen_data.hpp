#ifndef FLINT_GEN_DATA_HPP
#define FLINT_GEN_DATA_HPP
#include "implementation.hpp"

struct GenRandomImpl : OperationImplementation {
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		void generate_gpu_lazy() override {}
		void generate_gpu_eager() override {}
};

struct GenConstantImpl : OperationImplementation {
		template <typename T>
		void execute_cpu_type(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 T *__restrict__ result, size_t from,
						 size_t size);
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		void generate_gpu_lazy() override {}
		void generate_gpu_eager() override {}
};
struct GenArangeImpl : OperationImplementation {
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override;
		void generate_gpu_lazy() override {}
		void generate_gpu_eager() override {}
};
#endif
