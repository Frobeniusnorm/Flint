package datasets

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/stretchr/testify/assert"
	"math/rand"
	"testing"
)

func TestNewFakeDataset(t *testing.T) {
	dataset := NewFakeDataset(5, flint.Shape{10, 10}, 100)

	assert.Equal(t, "fake", dataset.Name)
	assert.Equal(t, flint.Shape{10, 10}, dataset.itemShape)
	assert.Equal(t, uint(5), dataset.categories)
	assert.Equal(t, uint(100), dataset.numItems)
}

func TestFakeDataset_Count(t *testing.T) {
	dataset := NewFakeDataset(5, flint.Shape{10, 10}, 100)

	assert.Equal(t, uint(100), dataset.Count())
}

func TestFakeDataset_Get(t *testing.T) {
	dataset := NewFakeDataset(5, flint.Shape{10, 10}, 100)
	entry := dataset.Get(1)

	assert.NotNil(t, entry.Label)
	assert.NotNil(t, entry.Data)

	data := flint.CalculateResult[float64](entry.Data.Node)
	label := flint.CalculateResult[int32](entry.Label.Node)

	assert.Equal(t, flint.Shape{10, 10}, data.Shape)
	assert.Equal(t, flint.Shape{1}, label.Shape)

	// assert data in [0, 1)
	for _, x := range data.Data.([]float64) {
		assert.GreaterOrEqual(t, x, float64(0))
		assert.Less(t, x, float64(1))
	}

	// assert label between 0 and 5 (categories)
	for _, x := range label.Data.([]int32) {
		assert.GreaterOrEqual(t, x, int32(0))
		assert.Less(t, x, int32(5))
	}
}

func TestFakeDataset_Collate(t *testing.T) {
	dataset := NewFakeDataset(5, flint.Shape{10, 10}, 100)
	entries := make([]FakeDatasetEntry, 32)
	for i := 0; i < len(entries); i++ {
		entries[i] = dataset.Get(uint(rand.Intn(100)))
	}

	collated := dataset.Collate(entries)
	assert.Equal(t, flint.Shape{32, 10, 10}, collated.Data.Node.GetShape())
	assert.Equal(t, flint.Shape{32}, collated.Label.Node.GetShape())
}
