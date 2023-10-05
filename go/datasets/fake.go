package datasets

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dl"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"log"
	"math/rand"
)

type FakeDataset struct {
	baseDataset
	itemShape  flint.Shape
	categories uint
	numItems   uint
}

type FakeDatasetEntry struct {
	Label dl.Tensor
	Data  dl.Tensor
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

func (d FakeDataset) Get(index uint) FakeDatasetEntry {
	if index >= d.numItems {
		panic("index out of bounds")
	}
	label := rand.Intn(int(d.categories))
	return FakeDatasetEntry{
		Label: dl.NewTensor(flint.CreateScalar(label, flint.F_INT32)),
		Data:  dl.NewTensor(flint.CreateGraphRandom(d.itemShape)),
	}
}

func (d FakeDataset) Collate(items []FakeDatasetEntry) FakeDatasetEntry {
	if len(items) <= 0 {
		log.Panicf("cannot collate items - invalid batch size (%d)", len(items))
	}
	labels := make([]dl.Tensor, len(items))
	images := make([]dl.Tensor, len(items))
	for idx, val := range items {
		labels[idx] = val.Label
		images[idx] = val.Data
	}
	return FakeDatasetEntry{
		Label: TrivialCollate(labels),
		Data:  TrivialCollate(images),
	}
}

func (d FakeDataset) String() string {
	return fmt.Sprintf("FakeDataset (%s) with %d items", d.Name, d.Count())
}
