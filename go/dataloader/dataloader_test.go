package dataloader

import (
	"github.com/Frobeniusnorm/Flint/go/datasets"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestDataloader_Count(t *testing.T) {
	dataset := datasets.NewFakeDataset(10, flint.Shape{10, 10}, 100)

	t.Run("drop last", func(t *testing.T) {
		trainDataloader := NewDataloader[datasets.FakeDatasetEntry](dataset, 32, true)
		assert.Equal(t, trainDataloader.Count(), uint(3))
	})

	t.Run("don't drop last", func(t *testing.T) {
		trainDataloader := NewDataloader[datasets.FakeDatasetEntry](dataset, 32, false)
		assert.Equal(t, trainDataloader.Count(), uint(4))
	})
}

func TestNewDataloader(t *testing.T) {
	dataset := datasets.NewFakeDataset(10, flint.Shape{10, 10}, 100)
	dl := NewDataloader[datasets.FakeDatasetEntry](dataset, 32, false)
	assert.Equal(t, dl.dataset, dataset)
	assert.Equal(t, dl.batchSize, 32)
	assert.Equal(t, dl.dropLast, false)
	assert.NotNil(t, dl.sampler)
	assert.Nil(t, dl.batchSampler)
	assert.Equal(t, uint(2), dl.prefetchFactor)
	assert.Equal(t, uint(0), dl.prevWorker)
	assert.Len(t, dl.remainingIndices, int(dataset.Count()-uint(64)))
	assert.Equal(t, uint(1), dl.numWorkers)
	assert.Len(t, dl.workerChannels, int(dl.numWorkers))
	assert.Len(t, dl.workerData, int(dl.numWorkers))
	assert.NotNil(t, dl.closeChan)

}

// FIXME: deadlocks even on a single worker!
func TestDataloader_Next(t *testing.T) {
	dataset := datasets.NewFakeDataset(10, flint.Shape{10, 10}, 10)
	dl := NewDataloader[datasets.FakeDatasetEntry](dataset, 3, false)

	t.Run("happy case", func(t *testing.T) {
		next, err := dl.Next()
		assert.NoError(t, err)
		assert.IsType(t, datasets.FakeDatasetEntry{}, next)
	})

	t.Run("last batch", func(t *testing.T) {
		_, err := dl.Next()
		assert.NoError(t, err)
		_, err = dl.Next()
		assert.NoError(t, err)
	})

	t.Run("keep going after last batch", func(t *testing.T) {
		_, err := dl.Next()
		assert.ErrorIs(t, err, Done)
	})
}
