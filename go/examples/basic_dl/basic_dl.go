package main

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dl/layers"
	losses "github.com/Frobeniusnorm/Flint/go/dl/loss"
	"github.com/Frobeniusnorm/Flint/go/dl/optimize"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

func main() {
	flint.Init(flint.BACKEND_BOTH)
	flint.SetLoggingLevel(flint.INFO)

	//trainDataset, testDataset, err := datasets.NewMnistDataset(path.Join("..", "..", "_data", "mnist"))
	//if err != nil {
	//	panic(err)
	//}

	//trainDataloader := dataloader.NewDataloader(trainDataset, 64, true)
	//testDataloader := dataloader.NewDataloader(testDataset, 32, true)

	// TODO: transform 28x28x3 images to 28*28*1 and then 784 linear

	model := layers.NewSequential(
		layers.NewFlatten(),
		layers.NewIdentity(),
		layers.NewFullyConnected(784, 32),
		layers.NewFullyConnected(32, 16),
		layers.NewRelu(),
		layers.NewFullyConnected(16, 10),
		layers.NewRelu(),
	)
	model.Train()

	//model := layers.NewFullyConnected(4, 2)
	fmt.Println(model)

	optim := optimize.NewSgd(model.Parameters(true), 1e-3)
	crit := losses.CrossEntropyLoss

	//for i := uint(0); i < testDataloader.Count(); i++ {
	for i := uint(0); i < 100; i++ {
		fmt.Println(i)
		//batch := testDataloader.Next()
		data := layers.NewTensor(flint.CreateGraphRandom(flint.Shape{64, 784}))
		labels := flint.CreateGraphConstant(10, flint.Shape{64, 10}, flint.F_INT32)
		flint.StartGradientContext()
		output := model.Forward(data)
		loss := layers.NewTensor(crit(output.Node, labels))
		flint.StopGradientContext()
		optim.Step(loss)
		flint.OptimizeMemory(loss.Node)
		flint.FreeGraph(loss.Node)
	}

	flint.Cleanup()
}
