package layers

import (
	"github.com/Frobeniusnorm/Flint/go/flint_old"
)

type Sigmoid struct{}

func NewSigmoid() Sigmoid {
	return Sigmoid{}
}

func (l Sigmoid) Parameters(_ bool) []flint_old.Parameter {
	return []flint_old.Parameter{}
}

func (l Sigmoid) Forward(x flint_old.Tensor) flint_old.Tensor {
	// FIXME: what to do with x? call x.Close?
	// Or change the flint_old in a way that it is reused with the result of this layer?
	exp := x.Neg().Exp()
	res := flint_old.Scalar(int32(1)).Div(exp.Add(flint_old.Scalar(int32(1))))
	return res
}

func (l Sigmoid) TrainMode() {}

func (l Sigmoid) EvalMode() {}

func (l Sigmoid) String() string {
	return "Sigmoid()"
}
