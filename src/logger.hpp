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

#ifndef LOGGER_H
#define LOGGER_H
#include <iostream>
// 0 no logging, 1 only errors, 2 errors and warnings, 3 error, warnings and
// info, 4 verbose, 5 debugging messages for the library developers
static int logging_level = 5;
enum LOGTYPE { DEBUG, VERBOSE, INFO, ERROR, WARNING };
inline void log(LOGTYPE type, std::string msg) {
  using namespace std;
  switch (type) {
  case DEBUG:
    if (logging_level >= 5)
      cout << "\033[0;33m[\033[0;32mDEBUG\033[0;33m]\033[0m " << msg << "\n";
    break;
  case VERBOSE:
    if (logging_level >= 4)
      cout << "\033[0;33m[\033[0;35mVERBOSE\033[0;33m]\033[0m " << msg << "\n";
    break;
  case INFO:
    if (logging_level >= 3)

      cout << "\033[0;33m[\033[0;36mINFO\033[0;33m]\033[0m " << msg << "\n";
    break;
  case WARNING:
    if (logging_level >= 2)
      cout << "\033[0;33m[\033[0;33mWARNING\033[0;33m]\033[0m " << msg << "\n";
    break;
  case ERROR:
    if (logging_level >= 1)
      cout << "\033[0;33m[\033[1;31mERROR\033[0;33m]\033[0m " << msg << "\n";
    throw std::runtime_error("error occured: " + msg);
  }
}
inline void setLoggerLevel(int level) { logging_level = level; }
#endif
