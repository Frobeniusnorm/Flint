package layers

import (
	"github.com/Frobeniusnorm/Flint/go/flint_old"
)

type ReLU struct{}

func NewRelu() ReLU {
	return ReLU{}
}

func (relu ReLU) Parameters(_ bool) []flint_old.Parameter {
	return []flint_old.Parameter{}
}

func (relu ReLU) Forward(x flint_old.Tensor) flint_old.Tensor {
	return x.Maximum(flint_old.Scalar(int32(0)))
}

func (relu ReLU) TrainMode() {}

func (relu ReLU) EvalMode() {}

func (relu ReLU) String() string {
	return "ReLU()"
}
