/*
This package provides a way to comfortably turn the go comments and signatures into the html format Flint uses for documentation.

For a reference on the Go docstring convention, look at: [go doc]

[go doc]: https://tip.golang.org/doc/comment
*/
package main

import (
	"fmt"
	"go/ast"
	"go/doc"
	"go/parser"
	"go/token"
	"html/template"
	"os"
)

func main() {
	// TODO: use context and go/build to find matching go files
	//	var Default Context = defaultContext()
	//	build.

	// Create the AST by parsing src and test.
	// fset is filled automatically by mustParse
	fset := token.NewFileSet()
	files := []*ast.File{
		mustParse(fset, "types.go"),
		mustParse(fset, "examples/gauss.go"),
	}

	// Compute package documentation the selected source files.
	p, err := doc.NewFromFiles(fset, files, "github.com/Frobeniusnorm/Flint/go")
	if err != nil {
		panic(err)
	}

	html := buildHtml(p)
	fmt.Println(html)

	//	fmt.Printf("package %s - %s", p.Name, p.Doc)
	//	fmt.Printf("func %s - %s", p.Funcs[0].Name, p.Funcs[0].Doc)
	// fmt.Printf(" ⤷ example with suffix %q - %s", p.Funcs[0].Examples[0].Suffix, p.Funcs[0].Examples[0].Doc)
}

// mustParse parses a source file or fails
func mustParse(fset *token.FileSet, filename string) *ast.File {
	f, err := parser.ParseFile(fset, filename, nil, parser.ParseComments)
	if err != nil {
		panic(err)
	}
	return f
}

func buildHtml(p *doc.Package) string {
	tmpl, err := template.New("flint.ahtml").Parse("hi")
	if err != nil {
		panic(err)
	}

	err = tmpl.Execute(os.Stdout, p)
	if err != nil {
		panic(err)
	}

	return "test"
}
