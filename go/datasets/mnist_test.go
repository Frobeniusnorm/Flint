package datasets

import (
	"fmt"
	"path"
	"testing"
)

// TestNewMnistDataset tests if the dataset at a given path can be loaded
// checking with valid and invalid path
func TestNewMnistDataset(t *testing.T) {

	t.Run("happy path", func(t *testing.T) {
		validPath := path.Clean("../_data/mnist")
		trainDataset, testDataset, err := NewMnistDataset(validPath)
		if err != nil {
			t.Log(err)
			t.Fatalf("loading from a valid path should NOT fail!")
		}
		fmt.Println(trainDataset.Get(0))
		fmt.Println(testDataset.Get(5))
	})

	t.Run("invalid path", func(t *testing.T) {
		invalidPath := path.Clean("../_data/no_mnist")
		_, _, err := NewMnistDataset(invalidPath)
		if err == nil {
			t.Fatalf("should fail with invalid path")
		}
	})
}
