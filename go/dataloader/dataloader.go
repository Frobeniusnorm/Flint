package dataloader

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/datasets"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

// Dataloader is a custom type that wraps a dataset and allows to iterate it in batches
type Dataloader struct {
	dataset   datasets.IndexableDataset
	batchSize uint
	dropLast  bool
	prevIndex uint
}

// TODO: see here: https://stackoverflow.com/questions/35810674/in-go-is-it-possible-to-iterate-over-a-custom-type
// and here: https://www.reddit.com/r/golang/comments/6gvq2j/custom_range_iterators_in_go/

func NewDataloader(dataset datasets.IndexableDataset, batchSize uint, dropLast bool) Dataloader {
	return Dataloader{
		dataset:   dataset,
		batchSize: batchSize,
		dropLast:  dropLast,
		prevIndex: 0,
	}
}

// Count returns the number of batches in the Dataloader
func (dl *Dataloader) Count() uint {
	dsSize := dl.dataset.Count()
	fullBatches := dsSize / dl.batchSize
	if dl.dropLast {
		return fullBatches
	} else {
		return fullBatches + 1
	}
}

func (dl *Dataloader) Next() any {
	var data flint.GraphNode = dl.dataset.Get(dl.prevIndex)
	flint.Concat()

	dl.dataset.Get()
}

func (dl *Dataloader) String() string {
	return fmt.Sprintf("Dataloader(dataset:%s, batch size: %d, dropLast: %t)",
		dl.dataset, dl.batchSize, dl.dropLast)
}
