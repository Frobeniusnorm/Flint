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
