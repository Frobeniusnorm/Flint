/*
Package dataloader provides an iterable dataloader for any dataset implementing the simple interface.
The provided dataloader can prefetch data across multiple workers to assure there is no slowdown in the training loop.
Inspired by PyTorch's dataloader.

for more information on custom iteration see:
- https://pkg.go.dev/google.golang.org/api/iterator
- https://stackoverflow.com/questions/35810674/in-go-is-it-possible-to-iterate-over-a-custom-type
- https://www.reddit.com/r/golang/comments/6gvq2j/custom_range_iterators_in_go/
- https://stackoverflow.com/questions/14000534/what-is-most-idiomatic-way-to-create-an-iterator-in-go
- https://stackoverflow.com/questions/35810674/in-go-is-it-possible-to-iterate-over-a-custom-type
- https://www.reddit.com/r/golang/comments/6gvq2j/custom_range_iterators_in_go/
NOTE: not able to provide a channel iterator as there would be no communication back to the workers to read more data while iterating over the channel
*/
package dataloader

import (
	"errors"
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/datasets"
	"log"
)

var Done = errors.New("no more items in Dataloader")

// Dataloader is a custom type that wraps a dataset and allows to iterate it in batches
// see (for inspiration): https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
type Dataloader[T any] struct {
	// dataset from which to load the data.
	dataset datasets.Dataset[T]

	//  how many samples per batch to load
	batchSize uint

	// If set to true, drop the last incomplete batch, if the dataset size is not divisible by the batchSize.
	// If false and the size of dataset is not divisible by the batchSize, then the last batch will be smaller than batchSize
	dropLast bool

	// defines the strategy to draw samples from the dataset.
	// Must return the next dataset index for each element in the dataset.
	// If specified, shuffle must not be specified.
	sampler func(remainingIndices *[]uint) (index uint, err error)

	// like sampler, but returns a batch of indices at a time.
	// Mutually exclusive with batch_size, shuffle, sampler, and dropLast.
	batchSampler func(remainingIndices *[]uint, batchSize uint, dropLast bool) (indices []uint, err error)

	// collate is a function that turns multiple dataset entries int a batch of dataset entries
	collate func(data []T) T

	// a function called upon initialization of the worker
	workerInit func(id uint)

	// Number of batches loaded in advance by each worker.
	// 2 means there will be a total of 2 * num_workers batches prefetched across all workers.
	prefetchFactor uint

	prevWorker       uint
	remainingIndices []uint

	numWorkers     uint
	workerChannels []chan int
	workerData     []chan T
	closeChan      chan struct{} // empty struct so it doesn't allocate any memory

	//wg sync.WaitGroup
}

// NewDataloader initializes a single-threaded dataloader from a given dataset.
func NewDataloader[T any](dataset datasets.Dataset[T], batchSize uint, dropLast bool) Dataloader[T] {
	return NewDataloaderMulti(dataset, batchSize, dropLast, 1)
}

// NewDataloaderMulti initializes a multi-threaded dataloader from a given dataset.
func NewDataloaderMulti[T any](dataset datasets.Dataset[T], batchSize uint, dropLast bool, numWorkers uint) Dataloader[T] {
	// Make sure [prefetchFactor * numWorkers > batchSize] to avoid bottlenecks.
	prefetchFactor := uint(2) // TODO: turn into param

	res := Dataloader[T]{
		// required  basics
		dataset:   dataset,
		batchSize: batchSize,
		dropLast:  dropLast,
		// functions
		sampler:      linearSampler,
		batchSampler: nil,
		collate:      nil,
		workerInit:   nil,
		// worker information
		prefetchFactor:   prefetchFactor,
		prevWorker:       0,
		remainingIndices: make([]uint, dataset.Count()),
		// workers
		numWorkers:     numWorkers,
		workerChannels: make([]chan int, numWorkers),
		workerData:     make([]chan T, numWorkers),
		closeChan:      make(chan struct{}),
	}

	// fill the remaining indices
	for i := uint(0); i < dataset.Count(); i++ {
		res.remainingIndices[i] = i
	}

	// start the workers
	for i := uint(0); i < numWorkers; i++ {
		res.workerData[i] = make(chan T, prefetchFactor)
		res.workerChannels[i] = make(chan int)
		go res.worker(i)

		// make them prefetch a few batches
		for p := uint(0); p < res.prefetchFactor; p++ {
			err := res.sendBatchIndices(i)
			if err != nil {
				panic("dataset does not even fit a single batch!")
			}
		}
	}

	return res
}

func (dl *Dataloader[T]) sendBatchIndices(workerId uint) error {
	if dl.batchSampler != nil {
		indices, err := dl.batchSampler(&dl.remainingIndices, dl.batchSize, dl.dropLast)
		if err != nil {
			return err
		}
		for _, v := range indices {
			dl.workerChannels[workerId] <- int(v)
		}
		return nil
	} else {
		for i := uint(0); i < dl.batchSize; i++ {
			index, err := dl.sampler(&dl.remainingIndices)
			if err != nil && !dl.dropLast {
				return err
			}
			dl.workerChannels[workerId] <- int(index)
		}
		return nil
	}
}

func (dl *Dataloader[T]) worker(id uint) {
	//defer dl.wg.Done()
	var storage = NewQueue[T](dl.batchSize)
	for {
		select {
		case index, ok := <-dl.workerChannels[id]:
			if !ok {
				fmt.Printf("worker %d: taskChan closed\n", id)
				return
			}

			data := dl.dataset.Get(uint(index))
			storage.Enqueue(data)

			if storage.Length() == dl.batchSize {
				batch := storage.DequeueAll()
				collatedBatch := dl.dataset.Collate(batch)
				dl.workerData[id] <- collatedBatch
			}

		case <-dl.closeChan:
			log.Printf("closing worker: %d.\n", id)
			return
		}
	}
}

// Count returns the number of batches in the Dataloader
func (dl *Dataloader[T]) Count() uint {
	dsSize := dl.dataset.Count()
	fullBatches := dsSize / dl.batchSize
	if dl.dropLast {
		return fullBatches
	} else {
		return fullBatches + 1
	}
}

func (dl *Dataloader[T]) Next() (batch T, err error) {
	workerId := dl.prevWorker
	dl.prevWorker = (dl.prevWorker + 1) % dl.numWorkers

	data, ok := <-dl.workerData[workerId]
	if !ok {
		log.Println("idk")
	}

	err = dl.sendBatchIndices(workerId)
	if err != nil {
		return batch, err
	}

	// TODO: signal "Done"

	return data, nil
}

func (dl *Dataloader[T]) Close() {
	close(dl.closeChan)
	for _, ch := range dl.workerChannels {
		close(ch)
	}
	//dl.wg.Wait()
}

func (dl *Dataloader[T]) String() string {
	return fmt.Sprintf("Dataloader(dataset:%s, batch size: %d, dropLast: %t)",
		dl.dataset, dl.batchSize, dl.dropLast)
}
