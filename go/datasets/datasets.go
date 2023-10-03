package datasets

import "fmt"

type baseDataset struct {
	Name string
}

type DatasetEntry any

type Dataset interface {
	fmt.Stringer
	Count() uint
	Get(index uint) (value DatasetEntry)

	// Collate combines multiple dataset entries into a single batched dataset entry
	Collate(items []DatasetEntry) DatasetEntry
}

const eps float32 = 1e-5

func TrainTestSplitAbsolute(dataset Dataset, trainSize uint, testSize uint) (Dataset, Dataset) {
	return dataset, dataset // TODO (implement. don't forget all the assertions about size)
}

func TrainTestSplitRelative(dataset Dataset, trainPercentage float32, testPercentage float32) (Dataset, Dataset) {
	return dataset, dataset // TODO (implement. don't forget all the assertions about size)
}
