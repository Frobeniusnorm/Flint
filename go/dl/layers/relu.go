package layers

import (
	"github.com/Frobeniusnorm/Flint/go/dl"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type ReLU struct{}

func NewRelu() ReLU {
	return ReLU{}
}

func (relu ReLU) Parameters(_ bool) []dl.Parameter {
	return []dl.Parameter{}
}

func (relu ReLU) Forward(x dl.Tensor) dl.Tensor {
	res := flint.Maximum(x.Node, int32(0))
	return dl.NewTensor(res)
}

func (relu ReLU) TrainMode() {}

func (relu ReLU) EvalMode() {}

func (relu ReLU) String() string {
	return "ReLU()"
}
