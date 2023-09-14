package main

import (
	"flint"
	"fmt"
)

func main() {
	flint.Init(flint.BACKEND_BOTH)
	flint.SetLoggingLevel(flint.INFO)
	flint.SetEagerExecution(false)

	//img := flint.LoadImage("../flint.png")
	img := flint.CreateGraphRandom(flint.Shape{30, 30, 30})

	fmt.Println("img shape (beggingning:", flint.GetShape(img))

	imgShape := flint.GetShape(img)
	h, w, c := imgShape[0], imgShape[1], imgShape[2]

	var kernelData = []float32{
		1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
		1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0,
		1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
	}
	kernel := flint.CreateGraph(kernelData, flint.Shape{1, 3, 3, 1})

	// FIXME: transpose kinda broken
	//img = flint.Transpose(img, flint.Axes{1, 1, 1})

	for i := 0; i < 500; i++ {
		// add padding
		img = flint.Extend(
			img,
			flint.Shape{c, w + 2, h + 2},
			flint.Axes{0, 1, 1},
		)
		// gaussian blur
		img = flint.Reshape(img, flint.Shape{c, w + 2, h + 2, 1})

		fmt.Println("img shape (before conv:", flint.GetShape(img))
		fmt.Println("kernel shape:", flint.GetShape(kernel))

		img = flint.Convolve(img, kernel, flint.Stride{1, 1, 1})
		img = flint.Execute(img)
	}

	//img = flint.Transpose(img, flint.Axes{0, 1, 2})

	flint.StoreImage(img, "gauss.jpg", flint.JPEG)

	flint.Cleanup()
}
