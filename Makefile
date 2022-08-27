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

COMPILER = g++ -std=c++2a  -l OpenCL
ifeq ($(MODE),debug)
	COMPILER += -g -Og -Wall
	.DEFAULT_GOAL :=test/test
else
	COMPILER += -O3
	.DEFAULT_GOAL :=libflint.a
endif

# WILD CARDS FOR COMPILATION
H_SRCS := $(wildcard $(SRCDIR)/*.hpp)
C_SRCS := $(wildcard $(SRCDIR)/*.cpp)
C_OBJS := $(C_SRCS:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)

# TEST
test/test: libflint.a $(BUILDDIR)/test.o | $(BUILDDIR)
	$(COMPILER) -o $@ $(BUILDDIR)/test.o -L. -lflint
	test/test

$(BUILDDIR)/test.o: test/test.cpp flint.h flint.hpp 
	$(COMPILER) -c -o $(BUILDDIR)/test.o test/test.cpp

# THE ACTUAL LIBRARY
libflint.a: $(C_OBJS) | $(BUILDDIR)
	ar -rc $@ $(C_OBJS)

# THE OBJECT FILES
$(BUILDDIR)/%.o: src/%.cpp $(H_SRCS)
	$(COMPILER) -c -o $@ $<

# OTHER TARGETS
build:
	mkdir build

clean:
	rm build/*
	rm test/test
	rm libflint.a
