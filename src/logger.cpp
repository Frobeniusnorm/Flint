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
