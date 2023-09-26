# Go Bindings for Flint

This go library provides a Go wrapper for [Flint](https://github.com/Frobeniusnorm/Flint).
Specifically, the `flint.h` file is wrapped.
The rest of the library is written in pure C++ so wrapping it is not possible.
It has to be rewritten using the functions from `flint.h`

## Setup

Make sure to first build and install the core library as
described [here](https://github.com/Frobeniusnorm/Flint#readme).

## Developer Guide

More infos about the architecture and technical details,
as well as the developer guide can be found [here](./DEVELOPMENT.md)
