package layers

import (
	"github.com/Frobeniusnorm/Flint/go/tensor"
)

type ReLU struct{}

func NewRelu() ReLU {
	return ReLU{}
}

func (relu ReLU) Parameters(_ bool) []tensor.Parameter {
	return []tensor.Parameter{}
}

func (relu ReLU) Forward(x tensor.Tensor) tensor.Tensor {
	return x.Maximum(tensor.Scalar(int32(0)))
}

func (relu ReLU) TrainMode() {}

func (relu ReLU) EvalMode() {}

func (relu ReLU) String() string {
	return "ReLU()"
}
