package datasets

import (
	"github.com/Frobeniusnorm/Flint/go/dl/layers"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"math/rand"
)

type FakeDataset struct {
	baseDataset
	itemShape  flint.Shape
	categories uint
	numItems   uint
}

type FakeDatasetEntry struct {
	label int
	data  layers.Tensor
}

func NewFakeDataset(categories uint, itemShape flint.Shape, numItems uint) FakeDataset {
	return FakeDataset{
		baseDataset: baseDataset{Name: "fake"},
		itemShape:   itemShape,
		categories:  categories,
		numItems:    numItems,
	}
}

func (d FakeDataset) Count() uint {
	return d.numItems
}

func (d FakeDataset) Get(index uint) DatasetEntry {
	return FakeDatasetEntry{
		label: rand.Int(),
		data:  layers.NewTensor(flint.CreateGraphRandom(d.itemShape)),
	}
}
