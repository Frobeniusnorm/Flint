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

	data1 := []float32{1, 2, 3, 4}
	//data2 := []float32{4, 3, 2, 1}
	shape := []uint64{2, 2}
	var g1 = flint.CreateGraph[float32](data1, shape)
	fmt.Println(g1)
	//var g2 flint.GraphNode = flint.NewGraphNode(data2, 4, flint.FLOAT32, shape)
	//var mm flint.GraphNode = flint.executeGraph(flint.matmul(g1, g2))
	//var rd flint.ResultData := mm.resultData
	//var result float64 := rd.data

	//var img flint.Tensor = 3 //[flint.INT32, 3] = flint.LoadImage("../flint.png")
	//h, w, c int := img.GetShape()
	//img = img.Transpose()
	//
	//for i := 0; i < 500; i++ {
	//	// add left padding
	//	img = img.extend([c, w+1, h+1], [0,1,1])
	//	// gauss
	//	img = img.reshape(c, w+1, h+1, 1).convolve(kernel, 1,1,1)
	//	// undo padding
	//	// TODO: img = img.slice()
	//	img.Execute()
	//}
	//
	//// undo transpose
	//img = img.Transpose()
	//
	//flint.StoreImage(img, "flint.jpg", JPEG)

	flint.Cleanup()
}
