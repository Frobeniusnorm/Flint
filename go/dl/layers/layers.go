package layers

import "fmt"

type BaseLayer struct {
}

type Layer interface {
	fmt.Stringer
	Parameters(recurse bool) []Parameter
	Forward(x Tensor) Tensor
	Train()
	Eval()
}
