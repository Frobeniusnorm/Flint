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
		validPath := path.Dir("../_data/mnist")
		trainDataset, testDataset, err := NewMnistDataset(validPath)
		if err != nil {
			t.Fatalf("loading from a valid path should NOT fail!")
		}
		fmt.Println(trainDataset, testDataset)
	})

	t.Run("invalid path", func(t *testing.T) {
		invalidPath := path.Dir("../_data/no_mnist")
		_, _, err := NewMnistDataset(invalidPath)
		if err == nil {
			t.Fatalf("should fail with invalid path")
		}
	})
}
