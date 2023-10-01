package main

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dl/layers"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

func main() {
	flint.SetLoggingLevel(flint.INFO)

	//trainDataset, testDataset, err := datasets.NewMnistDataset(path.Join("..", "..", "_data", "mnist"))
	//if err != nil {
	//	panic(err)
	//}

	//trainDataloader := dataloader.NewDataloader(trainDataset, 64, true)
	//testDataloader := dataloader.NewDataloader(testDataset, 32, true)

	// TODO: transform 28x28x3 images to 28*28*1 and then 784 linear

	//model := layers.NewSequential(
	//	layers.NewFlatten(),
	//	layers.NewIdentity(),
	//	layers.NewFullyConnected(784, 32),
	//	layers.NewFullyConnected(32, 16),
	//	layers.NewRelu(),
	//	layers.NewFullyConnected(16, 10),
	//	layers.NewRelu(),
	//)

	model := layers.NewFullyConnected(32, 4)
	fmt.Println(model)

	//optim := optimize.NewSgd(model.Parameters(true), 1e-3)
	//crit := losses.CrossEntropyLoss

	//for i := uint(0); i < testDataloader.Count(); i++ {
	//batch := testDataloader.Next()
	data := layers.NewTensor(flint.CreateGraphRandom(flint.Shape{8, 32}))
	//labels := flint.CreateGraphArrange(flint.Shape{4}, 0)
	output := model.Forward(data)
	fmt.Println(flint.CalculateResult[float32](output.Node))
	//loss := layers.NewTensor(crit(output.Node, labels))
	//optim.Step(loss)
}
