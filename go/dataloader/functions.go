package dataloader

import (
	"math/rand"
)

// linearSampler picks the next index from the list of available indices.
// NOTE: the picked index will be removed from [remainingIndices] as a side effect!
func linearSampler(remainingIndices *[]uint) (nextIndex uint) {
	idc := *remainingIndices
	res := idc[0]
	idc = idc[1:]
	remainingIndices = &idc
	return res
}

// randomSample picks one random index from the list of available indices.
// NOTE: the picked index will be removed from [remainingIndices] as a side effect!
func randomSample(remainingIndices *[]uint) (nextIndex uint) {
	idc := *remainingIndices
	resIdx := rand.Intn(len(idc))
	res := idc[resIdx]
	idc[resIdx] = idc[len(idc)-1] // replace removed element with the last one
	idc = idc[:len(idc)-1]        // cut of last element
	remainingIndices = &idc
	return res
}

func linearBatchSampler(remainingSampler *[]uint) (nextIndices []uint) {
	return nil
}

func randomBatchSampler(remainingSampler *[]uint) (nextIndices []uint) {
	return nil
}
