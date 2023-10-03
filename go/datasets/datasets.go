package datasets

type baseDataset struct {
	Name string
}

type DatasetEntry any

type Dataset interface {
	Count() uint
	Get(index uint) (value DatasetEntry)
}

const eps float32 = 1e-5

func TrainTestSplitAbsolute(dataset Dataset, trainSize uint, testSize uint) (Dataset, Dataset) {
	return dataset, dataset // TODO (implement. don't forget all the assertions about size)
}

func TrainTestSplitRelative(dataset Dataset, trainPercentage float32, testPercentage float32) (Dataset, Dataset) {
	return dataset, dataset // TODO (implement. don't forget all the assertions about size)
}
