package dataloader

import "github.com/Frobeniusnorm/Flint/go/datasets"

// Dataloader is a custom type that wraps a dataset and allows to iterate it in batches
type Dataloader struct {
	dataset   datasets.IndexableDataset
	batchSize uint
	shuffle   bool
}

// TODO: see here: https://stackoverflow.com/questions/35810674/in-go-is-it-possible-to-iterate-over-a-custom-type
// and here: https://www.reddit.com/r/golang/comments/6gvq2j/custom_range_iterators_in_go/

func NewDataloader(dataset datasets.IndexableDataset, batchSize uint) Dataloader {
	return Dataloader{
		dataset:   dataset,
		batchSize: batchSize,
		shuffle:   false,
	}
}
