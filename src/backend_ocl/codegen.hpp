/* Copyright 2023 David Schwarzbeck
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#ifndef OCL_CODEGEN_HPP
#define OCL_CODEGEN_HPP
#define FLINT_DEBUG
#include "../../flint.h"
#include "../utils.hpp"
#include <list>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
std::string
generateCode(FGraphNode *node,
			 std::list<std::pair<FGraphNode *, std::string>> &parameters);
std::string generateEagerCode(FOperationType operation, FType res_type,
							  std::vector<FType> parameter_types,
							  std::string &kernel_name);
#endif
