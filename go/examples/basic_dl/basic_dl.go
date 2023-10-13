package main

import (
	"errors"
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dataloader"
	"github.com/Frobeniusnorm/Flint/go/datasets"
	"github.com/Frobeniusnorm/Flint/go/dl/layers"
	"github.com/Frobeniusnorm/Flint/go/dl/losses"
	"github.com/Frobeniusnorm/Flint/go/dl/optim"
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

	trainDataloader := dataloader.NewDataloader[datasets.MnistDatasetEntry](trainDataset, 3, 64, true, nil, 2, false)
	fmt.Println(trainDataloader)
	testDataloader := dataloader.NewDataloader[datasets.MnistDatasetEntry](testDataset, 3, 32, true, nil, 1, false)
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

	optim := optim.NewSgd(model.Parameters(true), 1e-3)
	scheduler := optim.NewStepLR(optim, 10, 0.1) // essentially multiply learning rate by 0.1 every 10 epochs

	for epoch := 0; epoch < 3; epoch++ {
		train(model, trainDataloader, optim)
		test(model, trainDataloader)
		scheduler.Step()
		trainDataloader.Reset()
		testDataloader.Reset()
	}

	flint.Cleanup()
}

// train for one epoch
func train(model layers.Layer, trainDl *dataloader.Dataloader[datasets.MnistDatasetEntry], optim optim.Optimizer) {
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
		loss := losses.CrossEntropyLoss(output, batch.Label)
		flint.StopGradientContext()
		optim.Step(loss)
		flint.OptimizeMemory(loss.Node)
		flint.FreeGraph(loss.Node)
	}
}

// test the model on the test dataset
func test(model layers.Layer, testDl *dataloader.Dataloader[datasets.MnistDatasetEntry]) {
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
		loss := losses.CrossEntropyLoss(output, batch.Label)
		log.Println("test loss:", loss)
	}
}
