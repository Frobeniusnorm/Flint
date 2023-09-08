<div align="center">
    <img src="flint.png" width="300" alt="Flint logo">
</div>

# Flint
Flint (_Frobeniusnorm's Lightweight Implementation of Neural Tensors_) is a Tensor math framework based on OpenCL, allowing you to use Tensor operations, which are semi-lazily computed by the library. It supports gradient calculations for its operations including a gradient calculation algorithm. Once finished it will contain example implementations of the most popular optimization algorithms, loss functions and Neural Layers for machine learning.

## Motivation ##
This library was created by the frustration resulting from the inflexibility of other tensor math frameworks and their target languages for large projects. To ensure a high portability Flint generates OpenCL kernels for acceleration devices like GPUs on the fly.
This allows Flint to execute on a large amount of devices. The main target language is C++, allowing users of Flint to mix development and production code, with no need to rewrite everything for a faster language after e.g. training a machine learning model in python.
Planed Language Bindings include Haskell, Scala and maybe Java.

## Concept ##
Just as other Tensor libraries, Flint collects operations in a graph and executes multiple operations at once in one of its backends when needed.
The Graph (i.e. all yet to be calculated indirect parent nodes to one needed leaf node) is executed when the user of the library requests the resulting data or another operation depends on the full result so that combining those operations becomes impossible (i.e. operations like matrix multiplication or reducing). The Framework allows you to choose between CPU (with a thread pool that executes the tensor operations in parallel) and GPU execution (which generates OpenCL kernels for the execution graph), or you can let the framework decide on certain heuristics which backend to use. It additionally supports eager execution, for e.g. efficient cpu calculation or debugging purposes.

The main library only contains the implementation of the backends, the C++ frontend with the Tensor class and the operations with automatic gradient calculations. There is an example implementation of often used deep learning algorithms and concepts in the `dl/` folder in work.

## Usage ##
For prebuilt versions for Linux see the releases page. However, if you want to build it you need (a recent version of) a C++ compiler, CMake and an installed OpenCL library. 
To build the library execute this inside of the project folder:
```
mkdir build;
cd build;
cmake ..;
make;
```
to install it run `make install`.
Then just run the test programs (`test` and `test_gradients`, optionally `benchmark`)  to test if the implementation works on your machine. 
After that you can include the library in your project with
```
#include <flint/flint.hpp> // for the C++ implementation
#include <flint/flint.h> // for the C implementation
```
and link it against `-lflint -lOpenCL`.

The systems it was tested on ran Arch Linux, Gentoo, Fedora and Void. It was tested on AMD GPUs (integrated and dedicated), integrated Intel CPUs and on an NVIDIA GPU. 
There is a html documentation in the `docs/` folder: [https://frobeniusnorm.github.io/Flint/](https://frobeniusnorm.github.io/Flint/). You can also build it yourself with the Makefile in the docs folder, the prerequisite is a working version of ghc. However usually the newest version of the documentation is already built in the repository and deployed to the above mentioned site.

## Dependencies ##
- OpenCL
- C++ Compiler with C++20 feature implementations like e.g. semaphores

## Attribution to included libraries ##
- Benchmarking: [plf::nanotimer](https://github.com/mattreecebentley/plf_nanotimer) (zlib Licence)
- Testing: [Doctest](https://github.com/doctest/doctest) (MIT Licence)
- Image loading and writing: [stb](https://github.com/nothings/stb) (Public Domain)
