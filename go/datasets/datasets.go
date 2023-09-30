package datasets

type Dataset struct {
	Name string
}

type CommonDataset interface {
	Count() uint
}

type IndexableDataset interface {
	CommonDataset
	Get(index uint) (value any)
}

type IterableDataset interface {
	CommonDataset
	/*
		for more information on custom iteration see:
		- https://pkg.go.dev/google.golang.org/api/iterator
		- https://stackoverflow.com/questions/35810674/in-go-is-it-possible-to-iterate-over-a-custom-type
		- https://www.reddit.com/r/golang/comments/6gvq2j/custom_range_iterators_in_go/
		- https://stackoverflow.com/questions/14000534/what-is-most-idiomatic-way-to-create-an-iterator-in-go
	*/
	Next() (value any, ok bool)
}

const eps float32 = 1e-5

func (d *Dataset) TrainTestSplitAbsolute(trainSize uint, testSize uint) (Dataset, Dataset) {
	return Dataset{}, Dataset{} // TODO
}

func (d *Dataset) TrainTestSplitRelative(trainPercentage float32, testPercentage float32) (Dataset, Dataset) {
	return Dataset{}, Dataset{} // TODO

}
