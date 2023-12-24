package layers

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
)

// TODO: add parameter dim (int) – A dimension along which Softmax will be computed (so every slice along dim will sum to 1).

// LogSoftmax applies the logarithm to the softmax of the input [x]
func LogSoftmax(x flint.Tensor) flint.Tensor {
	defer x.Close()
	res := Softmax(x)
	defer res.Close()
	return res.Log()
}

type LogSoftmaxLayer struct{}

func NewLogSoftmax() LogSoftmaxLayer {
	return LogSoftmaxLayer{}
}

func (l LogSoftmaxLayer) Parameters(_ bool) []flint.Parameter {
	return []flint.Parameter{}
}

func (l LogSoftmaxLayer) Forward(x flint.Tensor) flint.Tensor {
	return LogSoftmax(x)
}

func (l LogSoftmaxLayer) TrainMode() {}

func (l LogSoftmaxLayer) EvalMode() {}

func (l LogSoftmaxLayer) String() string {
	return "LogSoftmax()"
}
