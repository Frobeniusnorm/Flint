package dataloader

import (
	"errors"
	"math/rand"
)

// linearSampler picks the next index from the list of available indices.
// NOTE: the picked index will be removed from [remainingIndices] as a side effect!
func linearSampler(remainingIndices *[]uint) (nextIndex uint, err error) {
	if remainingIndices == nil {
		return nextIndex, errors.New("remainingIndices should not be empty")
	}
	idc := *remainingIndices
	if len(idc) == 0 {
		return nextIndex, Done
	}
	nextIndex = idc[0]
	idc = idc[1:]
	*remainingIndices = idc
	return nextIndex, nil
}

// randomSample picks one random index from the list of available indices.
// NOTE: the picked index will be removed from [remainingIndices] as a side effect!
func randomSampler(remainingIndices *[]uint) (nextIndex uint, err error) {
	if remainingIndices == nil {
		return nextIndex, errors.New("remainingIndices should not be empty")
	}
	idc := *remainingIndices
	if len(idc) == 0 {
		return nextIndex, Done
	}
	resIdx := rand.Intn(len(idc))
	nextIndex = idc[resIdx]
	idc[resIdx] = idc[len(idc)-1] // replace removed element with the last one
	idc = idc[:len(idc)-1]        // cut of last element
	*remainingIndices = idc
	return nextIndex, nil
}

// linearBatchSampler picks a batch of consecutive indices from the available indices.
// NOTE: the picked indices will be removed from [remainingIndices] as a side effect!
func linearBatchSampler(remainingIndices *[]uint, batchSize uint, dropLast bool) (nextIndices []uint, err error) {
	if remainingIndices == nil {
		return nextIndices, errors.New("remainingIndices should not be empty")
	}
	idc := *remainingIndices
	if len(idc) == 0 {
		return nextIndices, Done
	}
	if dropLast == false && uint(len(idc)) < batchSize {
		return nextIndices, Done
	}
	idxLimit := min(uint(len(idc)), batchSize)
	nextIndices = idc[:idxLimit]
	idc = idc[idxLimit:]
	*remainingIndices = idc
	return nextIndices, nil
}

// randomBatchSampler picks a batch of random indices from the available indices.
// NOTE: the picked indices will be removed from [remainingIndices] as a side effect!
func randomBatchSampler(remainingIndices *[]uint, batchSize uint, dropLast bool) (nextIndices []uint, err error) {
	if remainingIndices == nil {
		return nextIndices, errors.New("remainingIndices should not be empty")
	}
	idc := *remainingIndices
	if len(idc) == 0 {
		return nil, Done
	}
	if dropLast == false && uint(len(idc)) < batchSize {
		return nextIndices, Done
	}
	idxLimit := min(len(idc), int(batchSize))
	nextIndices = make([]uint, idxLimit)
	for i := 0; i < idxLimit; i++ {
		next, _ := randomSampler(&idc)
		nextIndices[i] = next
	}
	*remainingIndices = idc
	return nextIndices, nil
}
