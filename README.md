# Flint
Tensor math framework based on OpenCL

## WORK IN PROGRESS ##
There is no official first version yet

## Motivation ##
This library aims to become a suitable replacement for tensor execution frameworks by using OpenCL instead of CUDA and should
work therefor on all types of graphic cards.
Planed Language Bindings include Java and Scala and a C++ wrapper library.

## Concept ##
Just as other Tensor libraries, Flint collects operations in a graph and executes multiple operations at once in OpenCL when needed. 

## Usage ##
There is no first official version yet and i haven't implemented a deploy target yet. However if you want to build it you need GCC, Make and a installed OpenCL library. Then just run make and test if the implementation works on your machine with the test executable build in the test directory.
