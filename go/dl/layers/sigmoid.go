package layers

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type Sigmoid struct{}

func NewSigmoid() Sigmoid {
	return Sigmoid{}
}

func (l Sigmoid) Parameters(_ bool) []flint.Parameter {
	return []flint.Parameter{}
}

func (l Sigmoid) Forward(x flint.Tensor) flint.Tensor {
	// FIXME: what to do with x? call x.Close?
	// Or change the flint in a way that it is reused with the result of this layer?
	exp := x.Neg().Exp()
	res := flint.Scalar(int32(1)).Div(exp.Add(flint.Scalar(int32(1))))
	return res
}

func (l Sigmoid) TrainMode() {}

func (l Sigmoid) EvalMode() {}

func (l Sigmoid) String() string {
	return "Sigmoid()"
}
