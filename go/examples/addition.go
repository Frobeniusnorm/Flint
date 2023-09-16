package main

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go"
)

func main() {
	flint.Init(flint.BACKEND_BOTH)
	flint.SetLoggingLevel(flint.VERBOSE)
	flint.Logging(flint.INFO, "testing logger in go")
	flint.SetEagerExecution(false)

	shape := flint.Shape{2, 3}
	var x = flint.CreateGraphConstant[int32](1, shape)
	var y = flint.CreateGraphConstant[int32](6, shape)
	z := flint.Add(x, y)
	res := flint.CalculateResult[int64](z)
	fmt.Println("res", res)

	flint.FreeGraph(z)
	flint.Cleanup()
}
