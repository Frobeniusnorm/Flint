<div align="center">
<img src="https://github.com/Frobeniusnorm/Flint/blob/main/flint.png" width="300">
</div>

# Flint
Flint is a Tensor math framework based on OpenCL, allowing you to write Tensor operations, which are 
semi-lazily computed by the library.

## Motivation ##
This library aims to become a suitable replacement for existing CUDA based tensor execution frameworks by using OpenCL instead.
This allows Flint to be executed on a larger amount of devices. The main target language will be C++, allowing users of Flint to mix developement and production code, with no need to rewrite everything for a faster language after e.g. training a machine learning model in python.
Planed Language Bindings include Java, Scala and a C++ wrapper library.

## Concept ##
Just as other Tensor libraries, Flint collects operations in a graph and executes multiple operations at once in OpenCL when needed.
The Graph (i.e. all yet to be calculated indirect parent nodes to one needed leaf node) is executed when the user of the library requests the resulting data or another operation depends on the full result so that combining those operations becomes impossible (i.e. operations like matrix multiplication or reducing). The Framework allows you to choose between CPU or GPU execution, or you can let the framework decide on certain heuristics which backend to use.

## Usage ##
There is no first official version yet. However if you want to build it you need GCC (g++), Make and a installed OpenCL library. Then just run make and test if the implementation works on your machine with the test executable build in the test directory.
To build the documentation you need Sphinx.
