package datasets

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/stretchr/testify/assert"
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
	entry := dataset.Get(2) // FIXME: changing this index is causing issues!!
	// Maybe overflow from uint to int conversions??

	assert.NotNil(t, entry.Label)
	assert.NotNil(t, entry.Data)
	assert.Equal(t, flint.Shape{10, 10}, entry.Data.Node.GetShape())
	assert.Equal(t, flint.Shape{1}, entry.Label.Node.GetShape())

	data := flint.CalculateResult[float64](entry.Data.Node)
	label := flint.CalculateResult[int32](entry.Label.Node)
	t.Log(label)
	t.Log(data)

	assert.Equal(t, flint.Shape{10, 10}, data.Shape)
	assert.Equal(t, flint.Shape{1}, label.Shape)

	assert.Equal(t, flint.F_INT32, label.DataType)
	assert.Equal(t, flint.F_FLOAT64, data.DataType)

	// assert data in [0, 1)
	for _, x := range data.Data.([]float64) {
		assert.GreaterOrEqual(t, x, float64(0))
		assert.Less(t, x, float64(1))
	}

	// assert label between 0 and 5 (categories)
	for _, x := range label.Data.([]int32) {
		// FIXME: maybe also due to the flint expand calls etc.
		assert.GreaterOrEqual(t, x, int32(0))
		assert.LessOrEqual(t, x, int32(1))
	}
}

func TestFakeDataset_String(t *testing.T) {
	dataset := NewFakeDataset(5, flint.Shape{10, 10}, 100)

	assert.Equal(t, "FakeDataset (fake) with 100 items", dataset.String())
}

// Additional tests for other methods can be added similarly.
