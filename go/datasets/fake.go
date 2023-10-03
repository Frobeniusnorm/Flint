package datasets

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dl/layers"
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
	Label layers.Tensor
	Data  layers.Tensor
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

func (d FakeDataset) Get(_ uint) FakeDatasetEntry {
	class := rand.Intn(int(d.categories))
	label := flint.CreateGraph([]int{class}, flint.Shape{1}, flint.F_INT32)
	label = flint.Extend(label, flint.Shape{d.categories}, flint.Axes{uint(class)})
	data := flint.CreateGraphRandom(d.itemShape)
	return FakeDatasetEntry{
		Label: layers.NewTensor(label),
		Data:  layers.NewTensor(data),
	}
}

func (d FakeDataset) Collate(items []FakeDatasetEntry) FakeDatasetEntry {
	if len(items) <= 0 {
		log.Panicf("cannot collate items - invalid batch size (%d)", len(items))
	}
	newDataShape := flint.Shape{1}
	for _, val := range items[0].Data.Node.GetShape() {
		newDataShape = append(newDataShape, val)
	}
	label := flint.Reshape(items[0].Label.Node, flint.Shape{1, 1})
	data := flint.Reshape(items[0].Data.Node, newDataShape)
	items[0].Label.Close()
	items[0].Data.Close()
	for _, val := range items[1:] {
		label = flint.Concat(label, val.Label.Node, 0)
		data = flint.Concat(data, val.Data.Node, 0)
		val.Data.Close()
		val.Label.Close()
	}
	return FakeDatasetEntry{
		Label: layers.NewTensor(label),
		Data:  layers.NewTensor(data),
	}
}

func (d FakeDataset) String() string {
	return fmt.Sprintf("FakeDataset (%s) with %d items", d.Name, d.Count())
}
