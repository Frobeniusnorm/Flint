#ifndef FLINT_CPU_COMMON_HPP
#define FLINT_CPU_COMMON_HPP

#include "../../flint.h"
#include <vector>

struct CPUResultData {
		void *data;
		FType type;
		size_t num_entries;
		std::vector<size_t> shape;
};

#endif
