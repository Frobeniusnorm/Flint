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

#ifndef FLINT_NN_H
#define FLINT_NN_H
#include "flint.h"
struct FGradientRecord {
  // TODO
};
/**
 * Calculates the error gradients for each of the variables in
 * :c:var:`vars_to_watch` for the calculation of :c:var:`y` (which should
 * contain the :c:struct:`FResultData`) relative to its error :c:var:`error`.
 * The allocated FGradientRecord contains the corresponding gradients, see the
 * additional functions for retrieving it.
 */
FGradientRecord *fcalculateGradients(FGraphNode *y, FGraphNode *error,
                                     FGraphNode **vars_to_watch, int num_vars);
/**
 * Retrieves the error gradient for a variable calculated in the corresponding
 * FGradientRecord.
 */
FGraphNode *fgetErrorGradient(FGradientRecord *record, FGraphNode *variable);
#endif
