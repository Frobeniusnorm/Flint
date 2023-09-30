package datasets

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
	"math/rand"
)

type FakeDataset struct {
	Dataset
	itemShape  flint.Shape
	categories uint
	numItems   uint
}

type FakeDatasetEntry struct {
	label int
	data  flint.GraphNode
}

func NewFakeDataset(categories uint, itemShape flint.Shape, numItems uint) FakeDataset {
	return FakeDataset{
		Dataset:    Dataset{Name: "fake"},
		itemShape:  itemShape,
		categories: categories,
		numItems:   numItems,
	}
}

func (d *FakeDataset) Count() uint {
	return d.numItems
}

func (d *FakeDataset) Next() (value FakeDatasetEntry, ok bool) {
	return FakeDatasetEntry{
		label: rand.Int(),
		data:  flint.CreateGraphRandom(d.itemShape),
	}, true
}
