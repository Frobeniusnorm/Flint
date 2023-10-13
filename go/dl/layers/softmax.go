package layers

import (
	"github.com/Frobeniusnorm/Flint/go/dl"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type Softmax struct{}

func NewSoftmax() Softmax {
	return Softmax{}
}

func (l Softmax) Parameters(_ bool) []dl.Parameter {
	return []dl.Parameter{}
}

func (l Softmax) Forward(x dl.Tensor) dl.Tensor {
	// shift by the maximum value to avoid exploding values
	// thus avoiding inaccuracies when using floating point types.
	// x.Close() // FIXME: is this needed? I mean the caller could still hold the graph node refrence ..
	shifted := flint.Sub(x.Node, flint.Max[int32](x.Node))
	shifted = flint.Exp(shifted)
	div1 := shifted
	div2 := flint.ReduceSum(flint.Exp(shifted), len(shifted.GetShape())-1)
	res := flint.Divide(div1, div2)
	return dl.NewTensor(res)
}

func (l Softmax) TrainMode() {}

func (l Softmax) EvalMode() {}

func (l Softmax) String() string {
	return "Softmax()"
}
