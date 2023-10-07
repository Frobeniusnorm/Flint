package layers

import (
	"github.com/Frobeniusnorm/Flint/go/dl"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type Sigmoid struct{}

func NewSigmoid() Sigmoid {
	return Sigmoid{}
}

func (l Sigmoid) Parameters(_ bool) []dl.Parameter {
	return []dl.Parameter{}
}

func (l Sigmoid) Forward(x dl.Tensor) dl.Tensor {
	// FIXME: what to do with x? call x.Close?
	// Or change the tensor in a way that it is reused with the result of this layer?
	exp := flint.Exp(flint.Neg(x.Node))
	res := flint.Divide(int32(1), flint.Add(exp, int32(1)))
	return dl.NewTensor(res)
}

func (l Sigmoid) TrainMode() {}

func (l Sigmoid) EvalMode() {}

func (l Sigmoid) String() string {
	return "Sigmoid()"
}
