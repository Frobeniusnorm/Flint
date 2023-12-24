package layers

import (
	"github.com/Frobeniusnorm/Flint/go/flint_old"
)

// TODO: add parameter dim (int) – A dimension along which Softmax will be computed (so every slice along dim will sum to 1).

// LogSoftmax applies the logarithm to the softmax of the input [x]
func LogSoftmax(x flint_old.Tensor) flint_old.Tensor {
	defer x.Close()
	res := Softmax(x)
	defer res.Close()
	return res.Log()
}

type LogSoftmaxLayer struct{}

func NewLogSoftmax() LogSoftmaxLayer {
	return LogSoftmaxLayer{}
}

func (l LogSoftmaxLayer) Parameters(_ bool) []flint_old.Parameter {
	return []flint_old.Parameter{}
}

func (l LogSoftmaxLayer) Forward(x flint_old.Tensor) flint_old.Tensor {
	return LogSoftmax(x)
}

func (l LogSoftmaxLayer) TrainMode() {}

func (l LogSoftmaxLayer) EvalMode() {}

func (l LogSoftmaxLayer) String() string {
	return "LogSoftmax()"
}
