package main

import (
	"flint"
	"fmt"
)

func main() {
	flint.Init(flint.BACKEND_BOTH)
	flint.SetLoggingLevel(flint.VERBOSE)
	fmt.Println("initialized backend (GO):", flint.InitializedBackend())
	flint.Logging(flint.INFO, "testing logger in go")

	flint.SetEagerExecution(false)
	fmt.Println("is eager exec (GO):", flint.IsEagerExecution())

	shape := flint.Shape{2, 3}
	var x = flint.CreateGraphConstant[int32](1, shape)
	fmt.Println("x", x)

	var y = flint.CreateGraphConstant[int32](6, shape)
	fmt.Println("y", y)

	z := flint.Add(x, y)
	fmt.Println("z", z)

	res := flint.CalculateResult[int64](z)
	fmt.Println("res", res)

	flint.Cleanup()
}
