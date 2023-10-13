# Documentation

Currently, there are two ways to view the documentation for this Go project:

## Html generation

To create a documentation in similar style as in the C++ flint library, we need to generate the HTML ourselves.
The go source files in this directory serve exactly this purpose.
Using integrated language tools to create an Abstract Syntax tree, we can use the information in combination with HTML
templates to create the output files.
However, this means we need to find and process all comments and examples ourselves.

Note that this approach uses the same parser as `go doc`.
This means if we want code to show up in the results we have to closely follow
the [go comment convention]((https://tip.golang.org/doc/comment).

Also take a look here for tips and tricks: <https://pkg.go.dev/github.com/fluhus/godoc-tricks>

## Go doc / pkgsite

This is the simpler approach.
Using `pkgsite` we can create a documentation similar to what you find on <https://pkg.go.dev>.

Setup:

1. Install the site generator: `go install golang.org/x/pkgsite/cmd/pkgsite@latest`
2. Make sure your go installation directory is included in your `PATH`, i.e.: `export PATH="$PATH:$HOME/go/bin"`
3. Serve the documentation: `pkgsite .`

This site generator follows all the typical Conventions for [comments](https://tip.golang.org/doc/comment)
and [examples](https://go.dev/blog/examples).
