package main

import "flint"

func main() {
	flint.Init(flint.BACKEND_ONLY_CPU)
	flint.SetLoggingLevel(flint.DEBUG)

	//data1 := []float32{1,2,3,4}
	//data2 := []float64{4,3,2,1}
	//shape := []int{2,2}
	//var g1 flint.GraphNode := flint.NewGraphNode(data1, 4, flint.FLOAT32, shape)
	//var g2 flint.GraphNode := flint.NewGraphNode(data2, 4, flint.FLOAT32, shape)
	//var mm flint.GraphNode := flint.executeGraph(flint.matmul(g1, g2))
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
	//flint.Cleanup()
}
