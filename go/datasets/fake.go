package datasets

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
	"log"
	"math/rand"
)

type FakeDataset struct {
	baseDataset
	itemShape  wrapper.Shape
	categories uint
	numItems   uint
}

type FakeDatasetEntry struct {
	Label flint.Tensor
	Data  flint.Tensor
}

func NewFakeDataset(categories uint, itemShape wrapper.Shape, numItems uint) FakeDataset {
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
		Label: flint.Scalar(int32(label)),
		Data:  flint.Random(d.itemShape),
	}
}

func (d FakeDataset) Collate(items []FakeDatasetEntry) FakeDatasetEntry {
	if len(items) <= 0 {
		log.Panicf("cannot collate items - invalid batch size (%d)", len(items))
	}
	labels := make([]flint.Tensor, len(items))
	images := make([]flint.Tensor, len(items))
	for idx, val := range items {
		labels[idx] = val.Label
		images[idx] = val.Data
	}
	res := FakeDatasetEntry{
		Label: TrivialCollate(labels),
		Data:  TrivialCollate(images),
	}
	res.Label = res.Label.Flatten()
	return res
}

func (d FakeDataset) String() string {
	return fmt.Sprintf("FakeDataset (%s) with %d items", d.Name, d.Count())
}
