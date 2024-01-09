// this package provides a very simple example of basic wrapper functionality, where no memory management is needed
package main

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
	"log"
)

func main() {
	err := wrapper.Init(wrapper.BACKEND_BOTH)
	if err != nil {
		log.Fatal(err)
	}
	wrapper.SetLoggingLevel(wrapper.LOG_INFO)

	shape := wrapper.Shape{2, 3}
	x, err := wrapper.CreateGraphConstantInt(1, shape)
	y, err := wrapper.CreateGraphConstantInt(6, shape)
	z, err := wrapper.AddGraphGraph(x, y)
	z, err = wrapper.CalculateResult(z)
	if err != nil {
		log.Fatal(err)
	}
	res := wrapper.GetResultValue[wrapper.Int](z)
	fmt.Println("res", res)

	wrapper.FreeGraph(z)
	err = wrapper.Cleanup()
	if err != nil {
		log.Fatal(err)
	} else {
		fmt.Println("done")
		fmt.Println("res:", res)
	}
}
