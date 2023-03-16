# 	 Copyright 2022 David Schwarzbeck
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#        http://www.apache.org/licenses/LICENSE-2.0
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

SRCDIR=src
BUILDDIR=build
# creates and executes test target if set to "debug", change to enable production mode 
MODE=debug

.PHONY: debug production
COMPILER = clang++ -std=c++2a -gdwarf-4 -l OpenCL
ifeq ($(MODE),debug)
	COMPILER += -g -Og -Wall
	.DEFAULT_GOAL :=debug
else
	COMPILER += -O3
	.DEFAULT_GOAL :=production
endif

# WILD CARDS FOR COMPILATION
H_SRCS := $(wildcard $(SRCDIR)/*.hpp)
C_SRCS := $(wildcard $(SRCDIR)/*.cpp)
C_OBJS := $(C_SRCS:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)

debug: libflint.a test/test test/test_gradients test/benchmark
production: libflint.a test/benchmark 
# TEST
test/test: libflint.a $(BUILDDIR)/test.o | $(BUILDDIR)
	$(COMPILER) -o $@ $(BUILDDIR)/test.o -L. -lflint

test/test_gradients: libflint.a $(BUILDDIR)/test_gradients.o | $(BUILDDIR)
	$(COMPILER) -o $@ $(BUILDDIR)/test_gradients.o -L. -lflint

test/benchmark: libflint.a $(BUILDDIR)/benchmark.o | $(BUILDDIR)
	$(COMPILER) -o $@ $(BUILDDIR)/benchmark.o -L. -lflint


$(BUILDDIR)/test.o: test/test.cpp flint.h flint.hpp 
	$(COMPILER) -c -o $(BUILDDIR)/test.o test/test.cpp

$(BUILDDIR)/test_gradients.o: test/test_gradients.cpp test/grad_test_cases.hpp flint.h flint.hpp 
	$(COMPILER) -c -o $(BUILDDIR)/test_gradients.o test/test_gradients.cpp

$(BUILDDIR)/benchmark.o: test/benchmark.cpp flint.hpp
	$(COMPILER) -c -o $(BUILDDIR)/benchmark.o test/benchmark.cpp

# THE ACTUAL LIBRARY
libflint.a: $(C_OBJS) | $(BUILDDIR)
	ar -rc $@ $(C_OBJS)

# THE OBJECT FILES
$(BUILDDIR)/%.o: src/%.cpp $(H_SRCS) | $(BUILDDIR)
	$(COMPILER) -c -o $@ $<

# OTHER TARGETS
build:
	mkdir build

clean:
	rm build/*
	rm test/test
	rm libflint.a
	rm test/benchmark
