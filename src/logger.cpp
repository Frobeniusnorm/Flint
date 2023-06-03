/* Copyright 2022 David Schwarzbeck

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
  This file includes the implementation for a simple logger.
*/
#include "../flint.h"
#include <iostream>
#include <stdexcept>

static int logging_level = 5;
void flogging(FLogType type, const char *msg) {
  using namespace std;
  switch (type) {
  case F_DEBUG:
    if (logging_level >= 5)
      cout << "\033[0;33m[\033[0;32mDEBUG\033[0;33m]\033[0m " << msg
           << std::endl;
    break;
  case F_VERBOSE:
    if (logging_level >= 4)
      cout << "\033[0;33m[\033[0;35mVERBOSE\033[0;33m]\033[0m " << msg
           << std::endl;
    break;
  case F_INFO:
    if (logging_level >= 3)

      cout << "\033[0;33m[\033[0;36mINFO\033[0;33m]\033[0m " << msg
           << std::endl;
    break;
  case F_WARNING:
    if (logging_level >= 2)
      cout << "\033[0;33m[\033[0;33mWARNING\033[0;33m]\033[0m " << msg
           << std::endl;
    break;
  case F_ERROR:
    if (logging_level >= 1)
      cout << "\033[0;33m[\033[1;31mERROR\033[0;33m]\033[0m " << msg
           << std::endl;
    throw std::runtime_error(msg); 
  }
}
void fSetLoggingLevel(int level) { logging_level = level; }
