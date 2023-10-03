/*
Package dataloader provides a iterable dataloader for any dataset implementing the simple interface.
The provided dataloader can prefetch data across multiple workers to assure there is no slowdown in the training loop.
Inspired by PyTorch's dataloader.

NOTE: use channels and a goroutine to implement iterators basically!!!
for more information on custom iteration see:
- https://pkg.go.dev/google.golang.org/api/iterator
- https://stackoverflow.com/questions/35810674/in-go-is-it-possible-to-iterate-over-a-custom-type
- https://www.reddit.com/r/golang/comments/6gvq2j/custom_range_iterators_in_go/
- https://stackoverflow.com/questions/14000534/what-is-most-idiomatic-way-to-create-an-iterator-in-go
TODO: see here: https://stackoverflow.com/questions/35810674/in-go-is-it-possible-to-iterate-over-a-custom-type
and here: https://www.reddit.com/r/golang/comments/6gvq2j/custom_range_iterators_in_go/
*/
package dataloader

import (
	"errors"
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/datasets"
	"math/rand"
	"sync"
)

var Done = errors.New("no more items in Dataloader")

type Sampler struct {
	maxItems  uint
	prevIndex uint
}

func (s *Sampler) linearSample() (nextIndex uint) {
	return s.prevIndex + 1
}

func (s *Sampler) randomSample() (nextIndex uint) {
	return uint(rand.Intn(int(s.maxItems)))
}

// Dataloader is a custom type that wraps a dataset and allows to iterate it in batches
// see (for inspiration): https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
type Dataloader struct {
	// dataset from which to load the data.
	dataset datasets.Dataset

	//  how many samples per batch to load
	batchSize uint

	// If set to true, drop the last incomplete batch, if the dataset size is not divisible by the batchSize.
	// If false and the size of dataset is not divisible by the batchSize, then the last batch will be smaller than batchSize
	dropLast bool

	// convenience option to use a random sampler.
	shuffle bool

	// defines the strategy to draw samples from the dataset.
	// Must return the next dataset index for each element in the dataset.
	// If specified, shuffle must not be specified.
	sampler func() (index uint)

	// like sampler, but returns a batch of indices at a time.
	// Mutually exclusive with batch_size, shuffle, sampler, and dropLast.
	batchSampler func() (indices []uint)

	workerInit func()

	// Number of batches loaded in advance by each worker.
	// 2 means there will be a total of 2 * num_workers batches prefetched across all workers.
	prefetchFactor uint

	remainingIndices []uint

	numWorkers     uint
	workerChannels []chan int
	workerStorage  []Queue[datasets.DatasetEntry]
	dataChan       chan datasets.DatasetEntry
	closeChan      chan struct{} // empty struct so it doesn't allocate any memory
	wg             sync.WaitGroup
}

// NewDataloader initializes a single-threaded dataloader from a given dataset.
func NewDataloader(dataset datasets.Dataset, batchSize uint, dropLast bool) *Dataloader {
	return NewDataloaderMulti(dataset, batchSize, dropLast, 1)
}

// NewDataloaderMulti initializes a multi-threaded dataloader from a given dataset.
func NewDataloaderMulti(dataset datasets.Dataset, batchSize uint, dropLast bool, numWorkers uint) *Dataloader {
	// Make sure [prefetchFactor * numWorkers > batchSize] to avoid bottlenecks.
	prefetchFactor := uint(2) // TODO: turn into param

	res := Dataloader{
		dataset:   dataset,
		batchSize: batchSize,
		dropLast:  dropLast,

		prefetchFactor: prefetchFactor,

		numWorkers:     numWorkers,
		workerChannels: make([]chan int, numWorkers),
		workerStorage:  make([]Queue[datasets.DatasetEntry], numWorkers),
		dataChan:       make(chan datasets.DatasetEntry, batchSize),
		closeChan:      make(chan struct{}),

		remainingIndices: make([]uint, dataset.Count()),
	}

	// fill the remaining indices
	for i := uint(0); i < dataset.Count(); i++ {
		res.remainingIndices[i] = i
	}

	// start the workers
	for i := uint(0); i < numWorkers; i++ {
		res.workerStorage[i] = NewQueue[datasets.DatasetEntry](prefetchFactor)
		res.workerChannels[i] = make(chan int)
		go res.worker(i)

		// add some indices for prefetching
		for p := uint(0); p < prefetchFactor; p++ {
			res.workerChannels[i] <- 1 // TODO
		}
	}

	return &res
}

func (dl *Dataloader) worker(id uint) {
	defer dl.wg.Done()
	for {
		select {
		case index, ok := <-dl.workerChannels[id]:
			if !ok {
				fmt.Printf("worker %d: taskChan closed\n", id)
				return
			}
			data := dl.dataset.Get(uint(index))
			dl.dataChan <- data
		case <-dl.closeChan:
			return
		}
	}
}

// Count returns the number of batches in the Dataloader
func (dl *Dataloader) Count() uint {
	dsSize := dl.dataset.Count()
	fullBatches := dsSize / dl.batchSize
	if dl.dropLast {
		return fullBatches
	} else {
		return fullBatches + 1
	}
}

func (dl *Dataloader) Next() (datasets.DatasetEntry, error) {
	if dl.current >= len(dl.data) {
		return 0, false
	}

	dl.workerChannels[dl.current%dl.workers] <- dl.current
	dl.current++

	select {
	case data := <-dl.dataChan:
		return data, true
	case <-dl.closeChan:
		return 0, false
	}
}

func (dl *Dataloader) Close() {
	close(dl.closeChan)
	for _, ch := range dl.workerChannels {
		close(ch)
	}
	dl.wg.Wait()
	// ??? close(dl.dataChan)
}

func (dl *Dataloader) String() string {
	return fmt.Sprintf("Dataloader(dataset:%s, batch size: %d, dropLast: %t)",
		dl.dataset, dl.batchSize, dl.dropLast)
}
