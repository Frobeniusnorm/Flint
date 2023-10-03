package main

import (
	"errors"
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dataloader"
	"github.com/Frobeniusnorm/Flint/go/datasets"
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

	trainDataloader := dataloader.NewDataloader[datasets.MnistDatasetEntry](trainDataset, 2, true)
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
	model.Train()

	//model := layers.NewFullyConnected(4, 2)
	fmt.Println(model)

	optim := optimize.NewSgd(model.Parameters(true), 1e-3)
	crit := losses.CrossEntropyLoss

	for i := 0; ; i++ {
		batch, err := trainDataloader.Next()
		if errors.Is(err, dataloader.Done) {
			break
		}
		if err != nil {
			log.Println(err)
			break
		}
		if i%50 == 0 {
			fmt.Printf("iteration: %d", i)
			fmt.Println(flint.CalculateResult[int](batch.Label.Node))
		}
		flint.StartGradientContext()
		output := model.Forward(batch.Data)
		loss := layers.NewTensor(crit(output.Node, batch.Label.Node))
		flint.StopGradientContext()
		optim.Step(loss)
		flint.OptimizeMemory(loss.Node)
		flint.FreeGraph(loss.Node)
	}

	flint.Cleanup()
}
