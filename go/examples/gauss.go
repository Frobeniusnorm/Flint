package main

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go"
)

func main() {
	flint.Init(flint.BACKEND_BOTH)
	flint.SetLoggingLevel(flint.INFO)

	img := flint.LoadImage("../../flint.png")

	//img := flint.CreateGraphRandom(flint.Shape{30, 30, 3})
	fmt.Println("img shape (beginning):", flint.GetShape(img))

	imgShape := flint.GetShape(img)
	h, w, c := imgShape[0], imgShape[1], imgShape[2]

	var kernelData = []float32{
		1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
		1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0,
		1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
	}
	kernel := flint.CreateGraph(kernelData, flint.Shape{1, 3, 3, 1})

	// channel in first dim
	img = flint.Transpose(img, flint.Axes{2, 1, 0})

	for i := 0; i < 500; i++ {
		// add padding
		img = flint.Extend(
			img,
			flint.Shape{c, w + 2, h + 2},
			flint.Axes{0, 1, 1},
		)
		// gaussian blur
		img = flint.Reshape(img, flint.Shape{c, w + 2, h + 2, 1})
		img = flint.Convolve(img, kernel, flint.Stride{1, 1, 1})
		img = flint.Execute(img)
	}

	// channel back into last dim
	img = flint.Transpose(img, flint.Axes{2, 1, 0})

	flint.StoreImage(img, "./gauss.bmp", flint.BMP)

	flint.Cleanup()
}
