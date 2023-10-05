package main

import (
	"errors"
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dataloader"
	"github.com/Frobeniusnorm/Flint/go/datasets"
	"github.com/Frobeniusnorm/Flint/go/dl"
	"github.com/Frobeniusnorm/Flint/go/dl/layers"
	losses "github.com/Frobeniusnorm/Flint/go/dl/loss"
	"github.com/Frobeniusnorm/Flint/go/dl/optimize"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"log"
	"path"
)

func main() {
	flint.Init(flint.BACKEND_ONLY_GPU)
	flint.SetLoggingLevel(flint.INFO)

	trainDataset, testDataset, err := datasets.NewMnistDataset(path.Join("..", "..", "_data", "mnist"))
	if err != nil {
		panic(err)
	}

	trainDataloader := dataloader.NewDataloader[datasets.MnistDatasetEntry](trainDataset, 64, true)
	fmt.Println(trainDataloader)
	testDataloader := dataloader.NewDataloader[datasets.MnistDatasetEntry](testDataset, 32, true)
	fmt.Println(testDataloader)

	model := layers.NewSequential(
		layers.NewFlatten(),
		layers.NewIdentity(),
		layers.NewFullyConnected(784, 32),
		layers.NewFullyConnected(32, 16),
		layers.NewRelu(),
		layers.NewFullyConnected(16, 10),
		layers.NewRelu(),
	)
	fmt.Println(model)

	optim := optimize.NewSgd(model.Parameters(true), 1e-3)
	scheduler := optimize.NewStepLR(optim, 10, 0.1) // essentially multiply learning rate by 0.1 every 10 epochs

	for epoch := 0; epoch < 3; epoch++ {
		train(model, trainDataloader, optim)
		test(model, trainDataloader)
		scheduler.Step()
	}

	flint.Cleanup()
}

// train for one epoch
func train(model layers.Layer, trainDl dataloader.Dataloader[datasets.MnistDatasetEntry], optim optimize.Optimizer) {
	model.EvalMode()
	for {
		batch, err := trainDl.Next()
		if errors.Is(err, dataloader.Done) {
			break
		}
		if err != nil {
			log.Println(err)
			break
		}

		flint.StartGradientContext()
		output := model.Forward(batch.Data)
		loss := dl.NewTensor(losses.CrossEntropyLoss(output.Node, batch.Label.Node))
		flint.StopGradientContext()
		optim.Step(loss)
		flint.OptimizeMemory(loss.Node)
		flint.FreeGraph(loss.Node)

		break
	}
}

// test the model on the test dataset
func test(model layers.Layer, testDl dataloader.Dataloader[datasets.MnistDatasetEntry]) {
	model.EvalMode()
	for {
		batch, err := testDl.Next()
		if errors.Is(err, dataloader.Done) {
			break
		}
		if err != nil {
			log.Println(err)
			break
		}

		output := model.Forward(batch.Data)
		loss := dl.NewTensor(losses.CrossEntropyLoss(output.Node, batch.Label.Node))
		log.Println("test loss:", loss)
	}
}
