package layers

import "github.com/Frobeniusnorm/Flint/go/flint"

type ReLU struct{}

func NewRelu() ReLU {
	return ReLU{}
}

func (relu ReLU) Parameters(recurse bool) []Parameter {
	return []Parameter{}
}

func (relu ReLU) Forward(x Tensor) Tensor {
	res := flint.Max(x.Node, int32(0))
	return Tensor{Node: res}
}

func (relu ReLU) Train() {}

func (relu ReLU) Eval() {}

func (relu ReLU) String() string {
	return "ReLU()"
}
