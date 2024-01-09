// this package provides a use case a little bit more complicated.
// Same as gauss_wrapper.go but using the flint abstraction layer.
// This avoids memory management issues
package main

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
)

func main() {
	wrapper.SetLoggingLevel(wrapper.LOG_INFO)
	img := flint.FromNodeWithErr(wrapper.LoadImage("../../gopher.png"))
	imgShape := img.Shape()
	h, w, c := imgShape[0], imgShape[1], imgShape[2]
	fmt.Println("img shape (beginning):", imgShape)

	// channel in first dim
	img.Transpose_(wrapper.Axes{2, 1, 0})

	var kernelData = []wrapper.Float{
		1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
		1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0,
		1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
	}
	kernel := flint.CreateFromSliceAndShape(kernelData, wrapper.Shape{1, 3, 3, 1}, wrapper.FLOAT32)

	wrapper.IncreaseRefCounter(kernel.Node()) // TODO: this should be done automatically
	wrapper.IncreaseRefCounter(img.Node())

	for i := 0; i < 200; i++ {
		fmt.Println("iteration:", i)

		// add padding
		img.Extend_(
			wrapper.Shape{c, w + 2, h + 2},
			wrapper.Axes{0, 1, 1},
		)
		// gaussian blur
		img.Reshape_(wrapper.Shape{c, w + 2, h + 2, 1})
		img.Convolve_(kernel, wrapper.Stride{1, 1, 1})

		_, _ = wrapper.ExecuteGraph(img.Node())
		_, _ = wrapper.OptimizeMemory(img.Node())
	}
	fmt.Println("done")

	// channel back into last dim
	img.Transpose_(wrapper.Axes{2, 1, 0})

	_ = wrapper.StoreImage(img.Node(), "./gauss.jpg", wrapper.JPEG)

	wrapper.DecreaseRefCounter(kernel.Node())
	wrapper.DecreaseRefCounter(img.Node())
	wrapper.FreeGraph(kernel.Node())
	wrapper.FreeGraph(img.Node())
	_ = wrapper.Cleanup()
}
