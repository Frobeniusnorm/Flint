package main

import (
	"flint"
	"fmt"
)

func main() {
	flint.Init(flint.BACKEND_BOTH)
	flint.SetLoggingLevel(flint.VERBOSE)
	fmt.Println("initialized backend (GO):", flint.InitializedBackend())
	flint.Log(flint.INFO, "testing logger in go")

	flint.SetEagerExecution(false)
	fmt.Println("is eager exec (GO):", flint.IsEagerExecution())

	data1 := []float32{1.2, 2.0, 3.1, 4.2}
	shape := flint.Shape{2, 2}
	var x = flint.CreateGraph(data1, shape)
	fmt.Println("x", x)

	var constantValue float32 = 5.0
	var y = flint.CreateGraphConstant(constantValue, shape)
	fmt.Println("y", y)

	z := flint.Add2(x, flint.F_INT(5))
	fmt.Println("z", z)

	res := z.Result()
	fmt.Println(res)
	fmt.Println("res", res)

	flint.Cleanup()
}
