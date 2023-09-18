# Developers guide for the Flint go wrapper

## Documentation

There is [`go doc`](https://go.dev/blog/godoc), but maybe the documentation can also be added to work with the stuff
that's already in `/docs`

See:

- <https://go.dev/doc/comment>

## Discussion

### Logging

Use [go logger](https://pkg.go.dev/log) or FFI/cgo calls to flint logger?

## Notes

The following "modern" C features do not work with cgo, all code needs to be in "legacy" C

- importing like `#include <string>`. Must be `#include <string.h>`
- defining enums without
  a `typedef` ([see here](https://stackoverflow.com/questions/34323130/the-importance-of-c-enumeration-typedef-enum)
  or [here](https://stackoverflow.com/questions/1102542/how-to-define-an-enumerated-type-enum-in-c))
    - alternatively preprend the enum type with the `enum` keyword everywhere it is used.

## TODO

- [ ] error handling: Cgo sets errno. Use go handling mechanism afterwards
- [ ] documentation: both as a flint .ahtml file, as well as pkgsite (maybe even on <go.pkg.dev>)

## Error handling

## Memory management

In iterative programs such as [`gauss.go`](./examples/gauss.go) a lot of unneeded nodes will be included in the graph at
some point.
The function `OptimizeMemory` helps with this issue.
To make sure it does not free relevant nodes, there are the reference counter manipulation methods (`IncreaseRefCounter`
and `DecreaseRefCounter`).
Increase the ref counter manually for nodes you want to keep, then decrease the counter when they should be freed.

## Checking for memory safety

Compile your code (e.g. the examples) with `go build examples/gauss.go` and run them with `valgrind`

NOTE: valgrind will ALWAYS report some leaked error. this is due to the CGO overhead and has nothing to do with this
program.

## Resources

- <https://stackoverflow.com/questions/38381357/how-to-point-to-c-header-files-in-go>
- <https://gist.github.com/zchee/b9c99695463d8902cd33>
- <https://zchee.github.io/golang-wiki/cgo/>
- <https://shane.ai/posts/cgo-performance-in-go1.21/>
- <https://dave.cheney.net/2016/01/18/cgo-is-not-go>
- <https://stackoverflow.com/questions/72332541/how-do-i-make-golang-cgo-to-use-g-linker-for-a-c-static-library-with-extern>
  Projects similar to this one:

- [gorgonia](https://github.com/gorgonia/gorgonia)
- [tfgo](https://github.com/galeone/tfgo)
- [gobrain](https://github.com/goml/gobrain)
- [gonet](https://github.com/dathoangnd/gonet)
- [go-deep](https://github.com/patrikeh/go-deep)

## Issues

- c++ exception handling:
    - <https://artem.krylysov.com/blog/2017/04/13/handling-cpp-exceptions-in-go/>
    - <https://github.com/golang/go/issues/12516>
    - <https://stackoverflow.com/questions/63185359/how-to-catch-the-c-c-lib-exceptions-in-go-code>
 