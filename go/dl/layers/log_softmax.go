package layers

import (
	"github.com/Frobeniusnorm/Flint/go/dl"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

// TODO: add parameter dim (int) – A dimension along which Softmax will be computed (so every slice along dim will sum to 1).

// LogSoftmax applies the logarithm to the softmax of the input [x]
func LogSoftmax(x dl.Tensor) dl.Tensor {
	defer x.Close()
	res := Softmax(x)
	defer res.Close()
	return dl.NewTensor(flint.Log(res.Node))
}

type LogSoftmaxLayer struct{}

func NewLogSoftmax() LogSoftmaxLayer {
	return LogSoftmaxLayer{}
}

func (l LogSoftmaxLayer) Parameters(_ bool) []dl.Parameter {
	return []dl.Parameter{}
}

func (l LogSoftmaxLayer) Forward(x dl.Tensor) dl.Tensor {
	return LogSoftmax(x)
}

func (l LogSoftmaxLayer) TrainMode() {}

func (l LogSoftmaxLayer) EvalMode() {}

func (l LogSoftmaxLayer) String() string {
	return "LogSoftmax()"
}
