// this package provides a very simple example of basic flint functionality, where no memory management is needed
package main

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

func main() {
	flint.Init(flint.BACKEND_BOTH)
	flint.SetLoggingLevel(flint.INFO)

	shape := flint.Shape{2, 3}
	var x = flint.CreateGraphConstant(1, shape, flint.F_INT32)
	var y = flint.CreateGraphConstant(6, shape, flint.F_INT32)
	z := flint.Add(x, y)
	res := flint.CalculateResult[int64](z)
	fmt.Println("res", res)

	flint.FreeGraph(z)
	flint.Cleanup()
}
