<div align="center">
<img src="https://github.com/Frobeniusnorm/Flint/blob/main/flint.png" width="300">
</div>

# Flint
Flint (_Frobeniusnorms Lightweight Implementation of Neural Tensors_) is a Tensor math framework based on OpenCL, allowing you to use Tensor operations, which are semi-lazily computed by the library. Once finished it will support gradient calculations for its operations including a feature rich gradient calculation algorithm with example implementations of the most popular optimization algorithms and loss functions for machine learning.

## Motivation ##
This library was created by the frustration resulting from the inflexible implementations of other tensor math frameworks. To ensure a high portability Flint generates OpenCL kernels for accelration devices like GPUs on the fly.
This allows Flint to execute on a large amount of devices. The main target language will be C++, allowing users of Flint to mix developement and production code, with no need to rewrite everything for a faster language after e.g. training a machine learning model in python.
Planed Language Bindings include Java, Scala and maybe Haskell.

## Concept ##
Just as other Tensor libraries, Flint collects operations in a graph and executes multiple operations at once in one of its backends when needed.
The Graph (i.e. all yet to be calculated indirect parent nodes to one needed leaf node) is executed when the user of the library requests the resulting data or another operation depends on the full result so that combining those operations becomes impossible (i.e. operations like matrix multiplication or reducing). The Framework allows you to choose between CPU (with a thread pool that executes the tensor operations in parallel) or GPU execution (which generates OpenCL kernels for the execution graph), or you can let the framework decide on certain heuristics which backend to use. It additionally supports eager execution, for e.g. efficient gradient calculation or debugging purposes.

## Usage ##
There is no first official version yet. However if you want to build it you need the Clang compiler, Make and a installed OpenCL library. Then just run make and test if the implementation works on your machine with the test executable build in the test directory.

## Dependencies ##
- OpenCL
- OpenCL C++ headers
- Clang C++ Compiler with C++20 feature implementations like e.g. semaphores

## Attribution to included libraries ##
- Testing: [Doctest](https://github.com/doctest/doctest) (MIT Licence)
- Benchmarking: [plf::nanotimer](https://github.com/mattreecebentley/plf_nanotimer) (zlib Licence)
- Documentation: [Doxygen](https://github.com/doxygen/doxygen)
