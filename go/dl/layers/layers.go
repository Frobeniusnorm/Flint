package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type Tensor struct {
	node flint.GraphNode
}

type BaseLayer struct {
	trainable     bool
	trainingPhase bool
}

type Layer interface {
	fmt.Stringer
	Forward(x Tensor) Tensor
	Backward(x Tensor) Tensor
}
