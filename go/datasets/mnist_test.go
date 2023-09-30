package datasets

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"path"
	"testing"
)

// TestNewMnistDataset tests if the dataset at a given path can be loaded
// checking with valid and invalid path
func TestNewMnistDataset(t *testing.T) {

	t.Run("happy path", func(t *testing.T) {
		validPath := path.Clean("../_data/mnist")
		trainDataset, _, err := NewMnistDataset(validPath)
		if err != nil {
			t.Log(err)
			t.Fatalf("loading from a valid path should NOT fail!")
		}

		entry := trainDataset.Get(2)
		var data = flint.CalculateResult[int32](entry.data)
		fmt.Println("label:", entry.label)
		fmt.Println("data:", data)
		printImage(data.Data.([]int32), data.Shape)

		//fmt.Println(testDataset.Get(5))
	})

	t.Run("invalid path", func(t *testing.T) {
		invalidPath := path.Clean("../_data/no_mnist")
		_, _, err := NewMnistDataset(invalidPath)
		if err == nil {
			t.Fatalf("should fail with invalid path")
		}
	})

}

// (debugging utility)
func printImage(image []int32, shape flint.Shape) {
	height, width := shape[0], shape[1]
	for row := uint(0); row < height; row++ {
		for col := uint(0); col < width; col++ {
			pix := image[row*height+col]
			if pix == 0 {
				fmt.Print(" ")
			} else {
				fmt.Printf("%X", pix/16)
			}
		}
		fmt.Println()
	}
}
