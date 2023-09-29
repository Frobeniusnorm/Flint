package models

import "github.com/Frobeniusnorm/Flint/go/dl/layers"

type LeNet5 struct {
	layers.BaseLayer
}

func NewLeNet5() LeNet5 {
	return LeNet5{}
}
