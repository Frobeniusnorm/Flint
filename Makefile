SRCDIR = src
BUILDDIR = build
MODE = debug # or production, defines the compiler arguments


COMPILER = g++
ifeq ($(MODE),"debug")
COMPILER += -Og -Wall
else
COMPILER += -O3
endif

# WILD CARDS FOR COMPILATION
H_SRCS := $(wildcard $(SRCDIR)/*.hpp)
C_SRCS := $(wildcard $(SRCDIR)/*.cpp)
C_OBJS := $(C_SRCS:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)

# TEST, IT'S ALSO THE DEFAULT
test/test: $(C_OBJS) $(BUILDDIR)/test.o | $(BUILDDIR)
	g++ -o $@ $(C_OBJS)$(BUILDDIR)/test.o

$(BUILDDIR)/test.o: test/test.cpp
	$(COMPILER) -c -o $(BUILDDIR)/test.o test/test.cpp

# THE OBJECT FILES
$(BUILDDIR)/%.o: src/%.cpp $(C_SRCS)
	$(COMPILER) -c -o $@ $<

# OTHER TARGETS
build:
	mkdir build

clean:
	rm build/*
