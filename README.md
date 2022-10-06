<center>
<img src="https://github.com/Frobeniusnorm/Flint/blob/main/flint.png" alt="Flint Logo" height="200"/>
</center>

# Flint
Tensor math framework based on OpenCL

## WORK IN PROGRESS ##
There is no official first version yet

## Motivation ##
This library aims to become a suitable replacement for existing CUDA based tensor execution frameworks by using OpenCL instead.
This allows Flint to be executed on a larger amount of devices. The main target language will be C++, allowing users of Flint to mix developement and production code, with no need to rewrite everything for a faster language after e.g. training a machine learning model in python.
Planed Language Bindings include Java, Scala and a C++ wrapper library.

## Concept ##
Just as other Tensor libraries, Flint collects operations in a graph and executes multiple operations at once in OpenCL when needed. 

## Usage ##
There is no first official version yet. However if you want to build it you need GCC (g++), Make and a installed OpenCL library. Then just run make and test if the implementation works on your machine with the test executable build in the test directory.
