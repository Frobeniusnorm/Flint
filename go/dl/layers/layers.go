package layers

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type Tensor flint.GraphNode

type BaseLayer struct {
	trainable     bool
	trainingPhase bool
}

type Layer interface {
	Forward(x Tensor) Tensor
	Backward(x Tensor) Tensor
}
