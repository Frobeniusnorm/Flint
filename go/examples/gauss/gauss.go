// this package provides a use case a little bit more complicated.
// To avoid linear memory growth we need calls to optimize memory as well as managing the reference counter
package main

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"log"
)

func main() {
	flint.Init(flint.BACKEND_BOTH)
	flint.SetLoggingLevel(flint.INFO)

	img, err := flint.LoadImage("../../../flint.png")
	if err != nil {
		log.Fatal(err)
	}

	imgShape := img.GetShape()
	h, w, c := imgShape[0], imgShape[1], imgShape[2]
	fmt.Println("img shape (beginning):", imgShape)

	// channel in first dim
	img = flint.Transpose(img, flint.Axes{2, 1, 0})

	var kernelData = []float32{
		1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
		1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0,
		1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
	}
	kernel := flint.CreateGraph(kernelData, flint.Shape{1, 3, 3, 1}, flint.F_FLOAT32)

	flint.IncreaseRefCounter(kernel)
	flint.IncreaseRefCounter(img)

	for i := 0; i < 200; i++ {
		fmt.Println("iteration:", i)
		// add padding
		img = flint.Extend(
			img,
			flint.Shape{c, w + 2, h + 2},
			flint.Axes{0, 1, 1},
		)
		// gaussian blur
		img = flint.Reshape(img, flint.Shape{c, w + 2, h + 2, 1})
		img = flint.Convolve(img, kernel, flint.Stride{1, 1, 1})

		img = flint.ExecuteGraph(img)
		img = flint.OptimizeMemory(img)
	}
	fmt.Println("done")

	// channel back into last dim
	img = flint.Transpose(img, flint.Axes{2, 1, 0})

	flint.StoreImage(img, "./gauss.jpg", flint.JPEG)

	flint.DecreaseRefCounter(kernel)
	flint.DecreaseRefCounter(img)
	flint.FreeGraph(kernel)
	flint.FreeGraph(img)
	flint.Cleanup()
}
