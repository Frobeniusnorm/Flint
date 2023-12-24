# Backend for OpenCL execution

Each new operation should be written for eager and lazy execution (eager works like a normal compiler without knowledge about the concrete nodes that are executeed, whereas lazy execution is more equivalent to jitting where a concrete node with all its parameters should be executed).
For each case a code generation pipeline has to be implemented respectively in the `codegen.hpp`, in the functions `generateCode` and `generateEagerCode`.

The eager pipeline is a little bit more complicated:
Because no prior information about the types is available the backend usually generates a kernel for each type combination of parameters, 
this behaviour may be manipulated per Operation type in the file `oclimpl.cpp` in the function `eagerCompile`.
In `codegen.hpp` there is the possibility to write the concrete kernel body and the parameters of its decleration in two seperate places.
The passing of parameters should be integrated in the functions `pushParameterVals` (which pushes parameters for each operand, so the function is run per operand of the node) and `pushAdditionalVals` which pushes parameter for the complete node. 
