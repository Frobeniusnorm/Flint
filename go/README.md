# Go Bindings for Flint

## Setup

1. first build flint as described in the [main readme](../README.md)
2. change the default _cgo_ compiler ([reference](https://stackoverflow.com/questions/44856124/can-i-change-default-compiler-used-by-cgo)): `go env -w "CC=g++`
3. build the go lib


## Build

`go build -o ./build ./examples/gauss.go #gosetup`

## Roadmap

### Examples

- [ ] Gauss
- [ ] Mnist

### Interface

First start with the ones relevant for Gauss example

- [ ] flintInit
- [ ] Tensor
  - [ ] reshape
  - [ ] transpose
  - [ ] expand
  - [ ] execute

### Documentation

There is [`go doc`](https://go.dev/blog/godoc), but maybe the documentation can also be added to work with the stuff thats already in `/docs`

See:
- <https://go.dev/doc/comment>

## Discussion

### Logging

Use [go logger](https://pkg.go.dev/log) or FFI/cgo calls to flint logger?


## Resources

- <https://stackoverflow.com/questions/38381357/how-to-point-to-c-header-files-in-go>


## Notes

The following "modern" C features do not work with cgo, all code needs to be in "legacy" C
- importing like `#include <string>`. Must be `#include <string.h>`
- defining enums without a `typedef` ([see here](https://stackoverflow.com/questions/34323130/the-importance-of-c-enumeration-typedef-enum) or [here](https://stackoverflow.com/questions/1102542/how-to-define-an-enumerated-type-enum-in-c))
  - alternatively preprend the enum type with the `enum` keyword everywhere it is used.

## Issues:
- linking issues (find a way to link libflint.a or global `-lflint`)
  - https://stackoverflow.com/questions/44856124/can-i-change-default-compiler-used-by-cgo
- memory management with cgo
- https://dave.cheney.net/2016/01/18/cgo-is-not-go