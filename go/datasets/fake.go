package datasets

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/Frobeniusnorm/Flint/go/tensor"
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
	Label tensor.Tensor
	Data  tensor.Tensor
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
		Label: tensor.NewTensor(flint.CreateScalar(label)),
		Data:  tensor.NewTensor(flint.CreateGraphRandom(d.itemShape)),
	}
}

func (d FakeDataset) Collate(items []FakeDatasetEntry) FakeDatasetEntry {
	if len(items) <= 0 {
		log.Panicf("cannot collate items - invalid batch size (%d)", len(items))
	}
	labels := make([]tensor.Tensor, len(items))
	images := make([]tensor.Tensor, len(items))
	for idx, val := range items {
		labels[idx] = val.Label
		images[idx] = val.Data
	}
	res := FakeDatasetEntry{
		Label: TrivialCollate(labels),
		Data:  TrivialCollate(images),
	}
	res.Label.Node = flint.Flatten(res.Label.Node)
	return res
}

func (d FakeDataset) String() string {
	return fmt.Sprintf("FakeDataset (%s) with %d items", d.Name, d.Count())
}
