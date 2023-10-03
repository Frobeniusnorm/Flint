package dataloader

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/datasets"
	"github.com/stretchr/testify/assert"
	"path"
	"testing"
)

func TestDataloader_Count(t *testing.T) {
	trainDataset, _, err := datasets.NewMnistDataset(path.Join("..", "_data", "mnist"))
	if err != nil {
		t.Fatal(err.Error())
	}

	mnistSize := trainDataset.Count()
	assert.Equal(t, mnistSize, uint(60000))
	fmt.Println("mnist train size:")

	t.Run("drop last", func(t *testing.T) {
		trainDataloader := NewDataloader(&trainDataset, 64, true)
		dlSize := trainDataloader.Count()
		fmt.Println("dataloader size drop last:", dlSize)
		assert.Equal(t, dlSize, uint(937))
	})

	t.Run("don't drop last", func(t *testing.T) {
		trainDataloader := NewDataloader(&trainDataset, 64, false)
		dlSize := trainDataloader.Count()
		fmt.Println("dataloader size DONT drop last:", dlSize)
		assert.Equal(t, dlSize, uint(938))
	})
}

func TestNewDataloader(t *testing.T) {
	ds, _, err := datasets.NewMnistDataset(path.Join("..", "_data", "mnist"))
	if err != nil {
		t.Fatal(err.Error())
	}

	dl := NewDataloader(ds, 64, false)
	fmt.Println(dl)
}
