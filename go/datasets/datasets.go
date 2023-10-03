package datasets

import "fmt"

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
