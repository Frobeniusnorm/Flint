/* Copyright 2022 David Schwarzbeck

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FLINT_HPP
#define FLINT_HPP

/* flint.hpp
 * This is the C++ implementation of Flint
 *
 * The core class of the C++ implementation is Tensor which has a template that
 * describes the dimensionality and type of the Tensor. All C++ functions use
 * the underlying implementations in flint.h.
 */

#include "flint.h"
// includes the template and helper classes
#include "flint_helper.hpp"
// includes the 1 dimensional implementation
#include "flint_1.hpp"
// includes the n dimensional implementation
#include "flint_n.hpp"

#endif
