#ifndef FLINT_CPU_COMMON_HPP
#define FLINT_CPU_COMMON_HPP

#include "../../flint.h"
#include <vector>

struct CPUResultData {
		void *data; // the result data of the node 
		FType type; // the data type of the result data
    bool multi_use = false; // for internal memory management
		size_t num_entries; // total number of entries in data
		std::vector<size_t> shape; // the original shape of the node
};

#endif
