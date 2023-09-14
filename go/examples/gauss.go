package main

import (
	"flint"
	"fmt"
)

func main() {
	flint.Init(flint.BACKEND_BOTH)
	flint.SetLoggingLevel(flint.INFO)
	flint.SetEagerExecution(false)

	img := flint.LoadImage("../flint.jpg")
	fmt.Println(img)
	//shape := flint.GetShape(img)
	//h, w, c := shape[0], shape[1], shape[2]
	//
	//var kernelData = []float32{1.0, 2.0}
	//kernel := flint.CreateGraph(kernelData, flint.Shape{3, 3})
	//
	//img = flint.Transpose(img, flint.Axes{0, 1, 2})
	//
	//for i := 0; i < 500; i++ {
	//	img = flint.Extend(
	//		img,
	//		flint.Shape{c, w + 2, h + 2},
	//		flint.Axes{0, 1, 1},
	//	)
	//
	//	img = flint.Reshape(img, flint.Shape{c, w + 2, h + 2, 1})
	//	img = flint.Convolve(img, kernel, flint.Stride{1, 1, 1})
	//	// TODO: execute
	//}
	//
	//img = flint.Transpose(img, flint.Axes{0, 1, 2})
	//
	//flint.StoreImage(img, "gauss.jpg", flint.JPEG)
	//
	//flint.Cleanup()
}
