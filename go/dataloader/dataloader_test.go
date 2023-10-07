package dataloader

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/datasets"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/stretchr/testify/assert"
	"testing"
	"time"
)

func TestDataloader_Count(t *testing.T) {
	t.Run("drop last", func(t *testing.T) {
		dataset := datasets.NewFakeDataset(2, flint.Shape{10, 10}, 10)
		dataload := NewDataloader[datasets.FakeDatasetEntry](dataset, 3, 3, true, nil, 0, false)
		assert.Equal(t, dataload.Count(), uint(3))
	})

	t.Run("drop last edge case", func(t *testing.T) {
		dataset := datasets.NewFakeDataset(2, flint.Shape{10, 10}, 9)
		dataload := NewDataloader[datasets.FakeDatasetEntry](dataset, 3, 3, true, nil, 0, false)
		assert.Equal(t, dataload.Count(), uint(3))
	})

	t.Run("don't drop last", func(t *testing.T) {
		dataset := datasets.NewFakeDataset(2, flint.Shape{10, 10}, 10)
		dataload := NewDataloader[datasets.FakeDatasetEntry](dataset, 3, 3, false, nil, 0, false)
		assert.Equal(t, dataload.Count(), uint(4))
	})

	t.Run("drop last edge case", func(t *testing.T) {
		dataset := datasets.NewFakeDataset(2, flint.Shape{10, 10}, 9)
		dataload := NewDataloader[datasets.FakeDatasetEntry](dataset, 3, 3, false, nil, 0, false)
		assert.Equal(t, dataload.Count(), uint(3))
	})
}

func TestNewDataloader(t *testing.T) {
	dataset := datasets.NewFakeDataset(10, flint.Shape{10, 10}, 1000)
	numWorkers := uint(3)
	batchSize := uint(32)
	prefetchFactor := uint(2)
	dl := NewDataloader[datasets.FakeDatasetEntry](dataset, numWorkers, batchSize, false, nil, prefetchFactor, false)

	assert.Equal(t, dl.dataset, dataset)
	assert.Equal(t, batchSize, dl.batchSize)
	assert.Equal(t, false, dl.dropLast)
	assert.NotNil(t, dl.sampler)
	assert.Nil(t, dl.batchSampler)
	assert.Equal(t, prefetchFactor, dl.prefetchFactor)
	assert.Equal(t, uint(0), dl.prevWorker)
	assert.Len(t, dl.remainingIndices, int(dataset.Count()-(2*3*uint(32))))
	assert.Equal(t, numWorkers, dl.numWorkers)
	assert.Len(t, dl.workerChannels, int(dl.numWorkers))
	assert.Len(t, dl.workerData, int(dl.numWorkers))
	assert.NotNil(t, dl.closeChan)
}

func TestNewDataloaderFromSampler(t *testing.T) {
	dataset := datasets.NewFakeDataset(10, flint.Shape{10, 10}, 1000)
	numWorkers := uint(3)
	batchSize := uint(32)
	prefetchFactor := uint(2)
	dl := NewDataloaderFromSampler[datasets.FakeDatasetEntry](dataset, numWorkers, batchSize, false, nil, prefetchFactor, linearSampler)

	assert.Equal(t, dl.dataset, dataset)
	assert.Equal(t, batchSize, dl.batchSize)
	assert.Equal(t, false, dl.dropLast)
	assert.NotNil(t, dl.sampler)
	assert.Nil(t, dl.batchSampler)
	assert.Equal(t, prefetchFactor, dl.prefetchFactor)
	assert.Equal(t, uint(0), dl.prevWorker)
	assert.Len(t, dl.remainingIndices, int(dataset.Count()-(2*3*uint(32))))
	assert.Equal(t, numWorkers, dl.numWorkers)
	assert.Len(t, dl.workerChannels, int(dl.numWorkers))
	assert.Len(t, dl.workerData, int(dl.numWorkers))
	assert.NotNil(t, dl.closeChan)
}

func TestNewDataloaderFromBatchSampler(t *testing.T) {
	dataset := datasets.NewFakeDataset(10, flint.Shape{10, 10}, 1000)
	numWorkers := uint(3)
	batchSize := uint(32)
	prefetchFactor := uint(2)
	dl := NewDataloaderFromBatchSampler[datasets.FakeDatasetEntry](dataset, numWorkers, batchSize, linearBatchSampler, nil, prefetchFactor)

	assert.Equal(t, dl.dataset, dataset)
	assert.Equal(t, batchSize, dl.batchSize)
	assert.Equal(t, false, dl.dropLast)
	assert.Nil(t, dl.sampler)
	assert.NotNil(t, dl.batchSampler)
	assert.Equal(t, prefetchFactor, dl.prefetchFactor)
	assert.Equal(t, uint(0), dl.prevWorker)
	assert.Len(t, dl.remainingIndices, int(dataset.Count()-(2*3*uint(32))))
	assert.Equal(t, numWorkers, dl.numWorkers)
	assert.Len(t, dl.workerChannels, int(dl.numWorkers))
	assert.Len(t, dl.workerData, int(dl.numWorkers))
	assert.NotNil(t, dl.closeChan)
}

func TestNewDataloaderSmart(t *testing.T) {
	dataset := datasets.NewFakeDataset(10, flint.Shape{10, 10}, 1000)
	batchSize := uint(32)
	dl := NewDataloaderSmart[datasets.FakeDatasetEntry](dataset, batchSize, false)

	assert.Equal(t, dl.dataset, dataset)
	assert.Equal(t, batchSize, dl.batchSize)
	assert.Equal(t, true, dl.dropLast)
	assert.NotNil(t, dl.sampler)
	assert.Nil(t, dl.batchSampler)
	assert.Equal(t, dl.prefetchFactor, uint(4))
	assert.Equal(t, uint(0), dl.prevWorker)
	assert.Len(t, dl.remainingIndices, int(dataset.Count()-(4*4*uint(32))))
	assert.Equal(t, dl.numWorkers, uint(4))
	assert.Len(t, dl.workerChannels, int(dl.numWorkers))
	assert.Len(t, dl.workerData, int(dl.numWorkers))
	assert.NotNil(t, dl.closeChan)
}

func TestDataloader_Next(t *testing.T) {
	dataset := datasets.NewFakeDataset(10, flint.Shape{10, 10}, 5)
	numWorkers := uint(2)
	batchSize := uint(2)
	prefetchFactor := uint(1)
	// we have at least 9 batches, 6 of them will be prefetched at the start
	dl := NewDataloader[datasets.FakeDatasetEntry](dataset, numWorkers, batchSize, true, nil, prefetchFactor, false)

	time.Sleep(time.Millisecond * 100)
	fmt.Println("happy case")
	next, err := dl.Next()
	assert.NoError(t, err)
	assert.IsType(t, datasets.FakeDatasetEntry{}, next)

	time.Sleep(time.Millisecond * 100)
	fmt.Println("last batch")
	_, err = dl.Next()
	assert.NoError(t, err)

	time.Sleep(time.Millisecond * 100)
	fmt.Println("keep going after last")
	_, err = dl.Next()
	assert.ErrorIs(t, err, Done)
	_, err = dl.Next()
	assert.ErrorIs(t, err, Done)
}

func TestDataloader_Reset(t *testing.T) {
	dataset := datasets.NewFakeDataset(10, flint.Shape{10, 10}, 10)
	numWorkers := uint(1)
	batchSize := uint(3)
	prefetchFactor := uint(0)
	dl := NewDataloader[datasets.FakeDatasetEntry](dataset, numWorkers, batchSize, true, nil, prefetchFactor, false)

	// do some calls to change the internal state
	_, _ = dl.Next()
	dl.workerData = nil
	dl.prevWorker = 500
	assert.Len(t, dl.remainingIndices, 7)
	assert.Nil(t, dl.workerData)
	assert.Equal(t, uint(500), dl.prevWorker)

	// reset and see if it is fixed again
	dl.Reset()
	assert.Equal(t, uint(0), dl.prevWorker)
	assert.NotNil(t, dl.workerData)
	assert.Len(t, dl.remainingIndices, 10)
}
