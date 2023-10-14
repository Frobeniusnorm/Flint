package layers

import (
	"github.com/Frobeniusnorm/Flint/go/dl"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

// TODO: add parameter dim (int) – A dimension along which Softmax will be computed (so every slice along dim will sum to 1).

// Softmax applies the softmax function to the input [x]
func Softmax(x dl.Tensor) dl.Tensor {
	// shift by the maximum value to avoid exploding values
	// thus avoiding inaccuracies when using floating point types.
	defer x.Close() // FIXME: is this needed? I mean the caller could still hold the graph node reference...
	shifted := flint.Sub(x.Node, flint.Max(x.Node))
	shifted = flint.Exp(shifted)
	div1 := shifted
	div2 := flint.ReduceSum(flint.Exp(shifted), len(shifted.GetShape())-1)
	res := flint.Divide(div1, div2)
	return dl.NewTensor(res)
}

type SoftmaxLayer struct{}

func NewSoftmax() SoftmaxLayer {
	return SoftmaxLayer{}
}

func (l SoftmaxLayer) Parameters(_ bool) []dl.Parameter {
	return []dl.Parameter{}
}

func (l SoftmaxLayer) Forward(x dl.Tensor) dl.Tensor {
	return Softmax(x)
}

func (l SoftmaxLayer) TrainMode() {}

func (l SoftmaxLayer) EvalMode() {}

func (l SoftmaxLayer) String() string {
	return "Softmax()"
}
