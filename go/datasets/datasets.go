package datasets

type Dataset struct {
	Name     string
	path     string
	size     uint
	labelMap map[uint]string
	data     []DatasetEntry
}

type DatasetEntry struct {
	label any
	data  any
}

type IndexableDataset interface {
	Count() uint
	Get(index uint) DatasetEntry
}

type IterableDataset interface {
	Count() uint
	Iterate()
}

func New() Dataset {
	return Dataset{}
}

const eps float32 = 1e-5

func (d *Dataset) TrainTestSplitAbsolute(trainSize uint, testSize uint) (Dataset, Dataset) {
	return Dataset{}, Dataset{} // TODO
}

func (d *Dataset) TrainTestSplitRelative(trainPercentage float32, testPercentage float32) (Dataset, Dataset) {
	return Dataset{}, Dataset{} // TODO

}
