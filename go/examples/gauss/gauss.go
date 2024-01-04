// this package provides a use case a little bit more complicated.
// To avoid linear memory growth we need calls to wrapper.OptimizeMemory as well as managing the reference counter
package main

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
)

func main() {
	_ = wrapper.Init(wrapper.BACKEND_BOTH)
	wrapper.SetLoggingLevel(wrapper.LOG_INFO)

	img, _ := wrapper.LoadImage("../../gopher.png")

	imgShape := img.GetShape()
	h, w, c := imgShape[0], imgShape[1], imgShape[2]
	fmt.Println("img shape (beginning):", imgShape)

	// channel in first dim
	img, _ = wrapper.Transpose(img, wrapper.Axes{2, 1, 0})

	var kernelData = []wrapper.Float{
		1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
		1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0,
		1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
	}
	kernel, _ := wrapper.CreateGraphDataFloat(kernelData, wrapper.Shape{1, 3, 3, 1})

	wrapper.IncreaseRefCounter(kernel)
	wrapper.IncreaseRefCounter(img)

	for i := 0; i < 200; i++ {
		fmt.Println("iteration:", i)
		// add padding
		img, _ = wrapper.Extend(
			img,
			wrapper.Shape{c, w + 2, h + 2},
			wrapper.Axes{0, 1, 1},
		)
		// gaussian blur
		img, _ = wrapper.Reshape(img, wrapper.Shape{c, w + 2, h + 2, 1})
		img, _ = wrapper.Convolve(img, kernel, wrapper.Stride{1, 1, 1})

		img, _ = wrapper.ExecuteGraph(img)
		img, _ = wrapper.OptimizeMemory(img)
	}
	fmt.Println("done")

	// channel back into last dim
	img, _ = wrapper.Transpose(img, wrapper.Axes{2, 1, 0})

	_ = wrapper.StoreImage(img, "./gauss.jpg", wrapper.JPEG)

	wrapper.DecreaseRefCounter(kernel)
	wrapper.DecreaseRefCounter(img)
	wrapper.FreeGraph(kernel)
	wrapper.FreeGraph(img)
	_ = wrapper.Cleanup()
}
