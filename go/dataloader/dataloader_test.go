package dataloader

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/datasets"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"path"
	"testing"
)

type MockDataset struct {
	mock.Mock
}

func TestDataloader_Count(t *testing.T) {
	dataset, _, err := datasets.NewMnistDataset(path.Join("..", "_data", "mnist"))
	if err != nil {
		t.Fatal(err.Error())
	}

	mnistSize := dataset.Count()
	assert.Equal(t, mnistSize, uint(60000))
	fmt.Println("mnist train size:")

	t.Run("drop last", func(t *testing.T) {
		trainDataloader := NewDataloader(&dataset, 64, true)
		dlSize := trainDataloader.Count()
		fmt.Println("dataloader size drop last:", dlSize)
		assert.Equal(t, dlSize, uint(937))
	})

	t.Run("don't drop last", func(t *testing.T) {
		trainDataloader := NewDataloader(&dataset, 64, false)
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
	t.Log(ds)

	dl := NewDataloader(ds, 64, false)
	fmt.Println(dl)
}

func TestDataloader_Next(t *testing.T) {
	dataset, _, err := datasets.NewMnistDataset(path.Join("..", "_data", "mnist"))
	if err != nil {
		t.Fatal(err.Error())
	}

	mnistSize := dataset.Count()
	assert.Equal(t, mnistSize, uint(60000))
	fmt.Println("mnist train size:")

	t.Run("happy case", func(t *testing.T) {
		testObj := new(MockDataset)
		testObj.On("")
	})

	t.Run("last batch", func(t *testing.T) {

	})

	t.Run("keep going after last batch", func(t *testing.T) {

	})
}
