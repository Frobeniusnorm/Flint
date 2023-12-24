// this package provides a very simple example of basic wrapper functionality, where no memory management is needed
package main

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
)

func main() {
	wrapper.Init(wrapper.BACKEND_BOTH)
	wrapper.SetLoggingLevel(wrapper.LOG_INFO)

	shape := wrapper.Shape{2, 3}
	var x = wrapper.CreateGraphConstant(1, shape)
	var y = wrapper.CreateGraphConstant(6, shape)
	z := wrapper.Add(x, y)
	res := wrapper.CalculateResult[int64](z)
	fmt.Println("res", res)

	wrapper.FreeGraph(z)
	wrapper.Cleanup()
}
