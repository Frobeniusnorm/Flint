#include "../flint.hpp"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Testing Logging") { log(INFO, "Hallo Welt!"); }
