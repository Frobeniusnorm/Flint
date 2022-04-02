LIBS=-lOpenCL

test: main.o
	g++ -o test main.o $(LIBS)

main.o: main.cpp core.hpp flint.hpp gpubackend.hpp
	g++ -c -g main.cpp 

