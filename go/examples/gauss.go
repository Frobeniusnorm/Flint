package main

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go"
)

func main() {
	flint.Init(flint.BACKEND_BOTH)
	flint.SetLoggingLevel(flint.INFO)

	img := flint.LoadImage("../../flint.png")

	imgShape := flint.GetShape(img)
	h, w, c := imgShape[0], imgShape[1], imgShape[2]
	fmt.Println("img shape (beginning):", imgShape)

	// channel in first dim
	img = flint.Transpose(img, flint.Axes{2, 1, 0})

	var kernelData = []float32{
		1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
		1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0,
		1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
	}
	kernel := flint.CreateGraph(kernelData, flint.Shape{1, 3, 3, 1})

	flint.IncreaseRefCounter(kernel)

	for i := 0; i < 500; i++ {
		fmt.Println("iteration", i)
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
		// TODO: fix memory issue first
		img = flint.OptimizeMemory(img) // FIXME: causes seg fault!!
	}

	flint.DecreaseRefCounter(kernel)
	flint.FreeGraph(kernel)

	// channel back into last dim
	img = flint.Transpose(img, flint.Axes{2, 1, 0})

	flint.StoreImage(img, "./gauss.bmp", flint.BMP)

	flint.FreeGraph(img)
	flint.Cleanup()

	fmt.Println("done")
}
