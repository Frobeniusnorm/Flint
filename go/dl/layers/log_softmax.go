package layers

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/Frobeniusnorm/Flint/go/tensor"
)

// TODO: add parameter dim (int) – A dimension along which Softmax will be computed (so every slice along dim will sum to 1).

// LogSoftmax applies the logarithm to the softmax of the input [x]
func LogSoftmax(x tensor.Tensor) tensor.Tensor {
	defer x.Close()
	res := Softmax(x)
	defer res.Close()
	return tensor.NewTensor(flint.Log(res.node))
}

type LogSoftmaxLayer struct{}

func NewLogSoftmax() LogSoftmaxLayer {
	return LogSoftmaxLayer{}
}

func (l LogSoftmaxLayer) Parameters(_ bool) []tensor.Parameter {
	return []tensor.Parameter{}
}

func (l LogSoftmaxLayer) Forward(x tensor.Tensor) tensor.Tensor {
	return LogSoftmax(x)
}

func (l LogSoftmaxLayer) TrainMode() {}

func (l LogSoftmaxLayer) EvalMode() {}

func (l LogSoftmaxLayer) String() string {
	return "LogSoftmax()"
}
