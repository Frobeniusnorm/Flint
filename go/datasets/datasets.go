package datasets

type Dataset struct{}

type IndexableDataset interface {
	Count() uint
	Get(index uint) any
}

type IterableDataset interface {
	Count() uint
	Iterate()
}

const eps float32 = 1e-5

func (d *Dataset) TrainTestSplitAbsolute(trainSize uint, testSize uint) (Dataset, Dataset) {
	return Dataset{}, Dataset{} // TODO
}

func (d *Dataset) TrainTestSplitRelative(trainPercentage float32, testPercentage float32) (Dataset, Dataset) {
	return Dataset{}, Dataset{} // TODO

}
