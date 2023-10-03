package datasets

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dl/layers"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"log"
)

type baseDataset struct {
	Name string
}

type Dataset[T any] interface {
	fmt.Stringer
	Count() uint
	Get(index uint) (value T)

	// Collate combines multiple dataset entries into a single batched dataset entry
	Collate(items []T) T
}

const eps float32 = 1e-5

func TrainTestSplitAbsolute[T any](dataset Dataset[T], trainSize uint, testSize uint) (Dataset[T], Dataset[T]) {
	return dataset, dataset // TODO (implement. don't forget all the assertions about size)
}

func TrainTestSplitRelative[T any](dataset Dataset[T], trainPercentage float32, testPercentage float32) (Dataset[T], Dataset[T]) {
	return dataset, dataset // TODO (implement. don't forget all the assertions about size)
}

func TrivialCollate(items []layers.Tensor) layers.Tensor {
	if len(items) <= 0 {
		log.Panicf("cannot collate items - invalid batch size (%d)", len(items))
	}

	newShape := flint.Shape{1}
	for _, val := range items[0].Node.GetShape() {
		newShape = append(newShape, val)
	}

	var res = flint.Reshape(items[0].Node, newShape)
	items[0].Close()
	for _, val := range items[1:] {
		res = flint.Concat(res, flint.Reshape(val.Node, newShape), 0)
		val.Close()
	}
	return layers.NewTensor(res)
}
