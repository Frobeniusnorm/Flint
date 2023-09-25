package layers

import flint "github.com/Frobeniusnorm/Flint/go"

type Tensor flint.GraphNode

type BaseLayer struct {
	trainable     bool
	trainingPhase bool
}

type Layer interface {
	Forward(x Tensor) Tensor
	Backward(x Tensor) Tensor
}
