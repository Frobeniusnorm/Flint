package layers

import "fmt"

type BaseLayer struct {
	trainable  bool
	EnableGrad bool // Essentially the same as "training_phase". Grads are disabled while testing.
}

type Layer interface {
	fmt.Stringer
	Parameters(recurse bool) []Parameter
	Forward(x Tensor) Tensor
}
