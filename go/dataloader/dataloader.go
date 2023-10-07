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
	"math"
)

var Done = errors.New("no more items in Dataloader")

type Sampler func(remainingIndices *[]uint) (index uint, err error)

type BatchSampler func(remainingIndices *[]uint, batchSize uint, dropLast bool) (indices []uint, err error)

type WorkerInit func(id uint)

// Dataloader is a custom type that wraps a dataset and allows to iterate it in batches
// see (for inspiration): https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
//
// NOTE: even tough this implementation supports multiple workers, it is NOT thread safe.
// Calling Next() from different goroutines is strongly discouraged!
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
	sampler Sampler

	// like sampler, but returns a batch of indices at a time.
	// Mutually exclusive with batch_size, sampler, and dropLast.
	batchSampler BatchSampler

	// a function called upon initialization of the worker
	workerInit WorkerInit

	// Number of batches loaded in advance by each worker.
	// 2 means there will be a total of 2 * num_workers batches prefetched across all workers.
	prefetchFactor uint

	// number of available workers
	numWorkers uint

	// number of batches that have yet to be requested
	remainingBatches int

	// batches are collected in round-robin fashion.
	// prevWorker is used to determine the next worker to collect from
	// No need to synchronize as this is only used by a single thread!
	prevWorker uint

	// the available indices of the dataset still left in this run.
	// No need to synchronize as this is only used by a single thread!
	remainingIndices []uint

	// workers receive the indices they should load through this channel
	workerChannels []chan []uint

	// workers send the result back through this channel
	workerData []chan T
}

func NewDataloaderSmart[T any](
	dataset datasets.Dataset[T],
	batchSize uint,
	shuffle bool,
) *Dataloader[T] {
	var sampler Sampler
	if shuffle {
		sampler = randomSampler
	} else {
		sampler = linearSampler
	}

	// TODO find a system to infer some good values for these
	numWorkers := uint(4)
	prefetchFactor := uint(4)

	res := &Dataloader[T]{
		dataset:        dataset,
		batchSize:      batchSize,
		dropLast:       true,
		sampler:        sampler,
		batchSampler:   nil,
		workerInit:     nil,
		prefetchFactor: prefetchFactor,
		numWorkers:     numWorkers,
	}
	return res.init()
}

func NewDataloader[T any](
	dataset datasets.Dataset[T],
	numWorkers uint,
	batchSize uint,
	dropLast bool,
	workerInit WorkerInit,
	prefetchFactor uint,
	shuffle bool,
) *Dataloader[T] {
	var sampler Sampler
	if shuffle {
		sampler = randomSampler
	} else {
		sampler = linearSampler
	}
	res := &Dataloader[T]{
		dataset:        dataset,
		batchSize:      batchSize,
		dropLast:       dropLast,
		sampler:        sampler,
		batchSampler:   nil,
		workerInit:     workerInit,
		prefetchFactor: prefetchFactor,
		numWorkers:     numWorkers,
	}
	return res.init()
}

func NewDataloaderFromSampler[T any](
	dataset datasets.Dataset[T],
	numWorkers uint,
	batchSize uint,
	dropLast bool,
	workerInit WorkerInit,
	prefetchFactor uint,
	sampler Sampler,
) *Dataloader[T] {
	res := &Dataloader[T]{
		dataset:        dataset,
		batchSize:      batchSize,
		dropLast:       dropLast,
		sampler:        sampler,
		batchSampler:   nil,
		workerInit:     workerInit,
		prefetchFactor: prefetchFactor,
		numWorkers:     numWorkers,
	}
	return res.init()
}

func NewDataloaderFromBatchSampler[T any](
	dataset datasets.Dataset[T],
	numWorkers uint,
	batchSize uint,
	batchSampler BatchSampler,
	workerInit WorkerInit,
	prefetchFactor uint,
) *Dataloader[T] {
	res := &Dataloader[T]{
		dataset:        dataset,
		batchSize:      batchSize,
		dropLast:       false,
		sampler:        nil,
		batchSampler:   batchSampler,
		workerInit:     workerInit,
		prefetchFactor: prefetchFactor,
		numWorkers:     numWorkers,
	}
	return res.init()
}

// NewDataloaderMulti initializes a multi-threaded dataloader from a given dataset.
func (dl *Dataloader[T]) init() *Dataloader[T] {
	// worker stuff
	dl.prevWorker = 0
	dl.remainingIndices = make([]uint, dl.dataset.Count())
	dl.workerChannels = make([]chan []uint, dl.numWorkers)
	dl.workerData = make([]chan T, dl.numWorkers)
	dl.remainingBatches = int(dl.Count())

	// fill the remaining indices
	for i := uint(0); i < dl.dataset.Count(); i++ {
		dl.remainingIndices[i] = i
	}

	// start the workers
	for i := uint(0); i < dl.numWorkers; i++ {
		dl.workerData[i] = make(chan T, dl.prefetchFactor+10)          // buffer needed so worker isn't blocked and can still receive.
		dl.workerChannels[i] = make(chan []uint, dl.prefetchFactor+10) // buffer needed so worker isn't blocked and can still receive.
		go dl.worker(i)

		// make them prefetch a few batches
		for p := uint(0); p < dl.prefetchFactor; p++ {
			indices, err := dl.getNextIndicesBatch()
			if err != nil {
				panic("dataset does not even contain enough batches for preloading!")
			}
			dl.workerChannels[i] <- indices
		}
	}
	return dl
}

// Reset prepares the dataloader for a new runs after it is done
// NOTE: this is done by copying most of the internal datastructures and reinitializing.
// Therefore, be careful when calling this, to avoid performance issues.
// This method is supposed to be called at the end of each epoch
func (dl *Dataloader[T]) Reset() *Dataloader[T] {
	res := &Dataloader[T]{
		dataset:        dl.dataset,
		batchSize:      dl.batchSize,
		dropLast:       dl.dropLast,
		sampler:        dl.sampler,
		batchSampler:   dl.batchSampler,
		workerInit:     dl.workerInit,
		prefetchFactor: dl.prefetchFactor,
		numWorkers:     dl.numWorkers,
	}
	dl.Close()
	*dl = *res.init() // replace the pointer contents with the newly initialized dataloader
	return dl
}

// Count returns the number of batches in the Dataloader
func (dl *Dataloader[T]) Count() uint {
	dsSize := dl.dataset.Count()
	fullBatches := float64(dsSize) / float64(dl.batchSize)
	if dl.dropLast {
		return uint(math.Floor(fullBatches))
	} else {
		return uint(math.Ceil(fullBatches))
	}
}

// getNextIndicesBatch creates a list of indices for the next batch.
// Handles all the edge cases or drop last, batchSampler etc.
func (dl *Dataloader[T]) getNextIndicesBatch() ([]uint, error) {
	if dl.batchSampler != nil {
		indices, err := dl.batchSampler(&dl.remainingIndices, dl.batchSize, dl.dropLast)
		return indices, err
	} else {
		indices := make([]uint, 0, dl.batchSize)
		for i := uint(0); i < dl.batchSize; i++ {
			index, err := dl.sampler(&dl.remainingIndices)
			if err != nil {
				if dl.dropLast && len(indices) > 0 {
					break
				} else {
					return nil, err
				}
			}
			indices = append(indices, index)
		}
		return indices, nil
	}
}

func (dl *Dataloader[T]) worker(id uint) {
	fmt.Println("start worker:", id)
	for {
		select {
		case indices, ok := <-dl.workerChannels[id]:
			if !ok {
				close(dl.workerData[id])
				fmt.Println("shutdown worker:", id)
				return
			}

			batch := make([]T, len(indices))
			for i, index := range indices {
				batch[i] = dl.dataset.Get(index)
			}
			collatedBatch := dl.dataset.Collate(batch)
			dl.workerData[id] <- collatedBatch
		}
	}
}

func (dl *Dataloader[T]) Next() (batch T, err error) {
	if dl.remainingBatches <= 0 {
		return batch, Done
	}

	// select next worker in a round-robin fashion
	workerId := dl.prevWorker
	dl.prevWorker = (dl.prevWorker + 1) % dl.numWorkers

	// try to get and send next batch
	indices, err := dl.getNextIndicesBatch()
	if err != nil {
		close(dl.workerChannels[workerId])
	} else {
		dl.workerChannels[workerId] <- indices
	}

	// read the data from the worker (must be there, either prefetched or because we just sent the indices
	batch = <-dl.workerData[workerId]

	dl.remainingBatches--
	return batch, nil
}

func (dl *Dataloader[T]) Close() {
	for _, ch := range dl.workerChannels {
		close(ch)
	}
}

func (dl *Dataloader[T]) String() string {
	return fmt.Sprintf("Dataloader(dataset:%s, batch size: %d, dropLast: %t)",
		dl.dataset.String(), dl.batchSize, dl.dropLast)
}
