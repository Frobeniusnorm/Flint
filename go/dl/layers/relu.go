package layers

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type ReLU struct{}

func NewRelu() ReLU {
	return ReLU{}
}

func (relu ReLU) Parameters(_ bool) []flint.Parameter {
	return []flint.Parameter{}
}

func (relu ReLU) Forward(x flint.Tensor) flint.Tensor {
	return x.Maximum(flint.Scalar(int32(0)))
}

func (relu ReLU) TrainMode() {}

func (relu ReLU) EvalMode() {}

func (relu ReLU) String() string {
	return "ReLU()"
}
