package main

import (
	"github.com/Frobeniusnorm/Flint/go/dataloader"
	"github.com/Frobeniusnorm/Flint/go/datasets"
	"github.com/Frobeniusnorm/Flint/go/dl/layers"
	"github.com/Frobeniusnorm/Flint/go/dl/loss"
	"path"
)

func main() {
	trainDataset, testDataset, err := datasets.NewMnistDataset(path.Join("..", "..", "_data", "mnist"))
	if err != nil {
		panic(err)
	}

	trainDataloader := dataloader.NewDataloader(trainDataset, 64, true)
	testDataloader := dataloader.NewDataloader(testDataset, 32, true)

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

	//optim := optimize.NewSgd(1e-3)
	crit := loss.CrossEntropyLoss

	for i := uint(0); i < testDataloader.Count(); i++ {
		data, labels := testDataloader.Next()
		output := model.Forward(data)
		crit(output.Node, labels)
	}
}
