# Go Bindings for Flint

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

The following "modern" C features do not work with cgo
- importing like `#include <string>`. Must be `#include <string.h>`
- defining enums without a `typedef`
  - alternatively preprend the enum type with the `enum` keyword everywhere it is used.

## Issues:
- linking issues (find a way to link libflint.a or global `-lflint`)
- can't build the examples directory (cmake file broken)
- memory management with cgo
