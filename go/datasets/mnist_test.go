package datasets

import (
	"github.com/Frobeniusnorm/Flint/go/wrapper"
	"github.com/stretchr/testify/assert"
	"math/rand"
	"path"
	"testing"
)

func validPath() string {
	return path.Clean(path.Join("..", "_data", "mnist"))
}

// TestNewMnistDataset tests if the dataset at a given path can be loaded
// checking with valid and invalid path
func TestNewMnistDataset(t *testing.T) {

	t.Run("happy path", func(t *testing.T) {
		trainDataset, testDataset, err := NewMnistDataset(validPath())
		assert.NoError(t, err)
		assert.Equal(t, "mnist-train", trainDataset.Name)
		assert.Equal(t, "mnist-test", testDataset.Name)

	})

	t.Run("invalid path", func(t *testing.T) {
		invalidPath := path.Clean("../_data/no_mnist")
		_, _, err := NewMnistDataset(invalidPath)
		if err == nil {
			t.Fatalf("should fail with invalid path")
		}
	})

}

func TestMnistDataset_Count(t *testing.T) {
	trainDataset, testDataset, err := NewMnistDataset(validPath())
	assert.NoError(t, err)
	assert.Equal(t, uint(60000), trainDataset.size)
	assert.Equal(t, uint(10000), testDataset.size)
	assert.Equal(t, uint(60000), trainDataset.Count())
	assert.Equal(t, uint(10000), testDataset.Count())
}

func TestMnistDataset_Get(t *testing.T) {
	dataset, _, err := NewMnistDataset(validPath())
	assert.NoError(t, err)

	entry := dataset.Get(1)

	assert.NotNil(t, entry.Label)
	assert.NotNil(t, entry.Data)

	data := wrapper.CalculateResult[float64](entry.Data.node)
	label := wrapper.CalculateResult[int32](entry.Label.node)

	assert.Equal(t, wrapper.Shape{28, 28}, data.Shape)
	assert.Equal(t, wrapper.Shape{1}, label.Shape)

	// assert data in [0, 256)
	for _, x := range []float64(data.Data) {
		assert.GreaterOrEqual(t, x, float64(0))
		assert.Less(t, x, float64(256))
	}

	// assert label between 0 and 9 (categories)
	for _, x := range []int32(label.Data) {
		assert.GreaterOrEqual(t, x, int32(0))
		assert.Less(t, x, int32(10))
	}
}

func TestMnistDataset_Collate(t *testing.T) {
	dataset, _, err := NewMnistDataset(validPath())
	assert.NoError(t, err)
	entries := make([]MnistDatasetEntry, 32)
	for i := 0; i < len(entries); i++ {
		entries[i] = dataset.Get(uint(rand.Intn(100)))
	}

	collated := dataset.Collate(entries)
	assert.Equal(t, wrapper.Shape{32, 28, 28}, collated.Data.node.GetShape())
	assert.Equal(t, wrapper.Shape{32}, collated.Label.node.GetShape())
}
