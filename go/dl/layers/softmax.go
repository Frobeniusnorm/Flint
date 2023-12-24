package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint_old"
)

// TODO: add parameter dim (int) – A dimension along which Softmax will be computed (so every slice along dim will sum to 1).

// Softmax applies the softmax function to the input [x]
func Softmax(x flint_old.Tensor) flint_old.Tensor {
	fmt.Println("Softmax")
	// shift by the maximum value to avoid exploding values
	// thus avoiding inaccuracies when using floating point types.
	defer x.Close() // FIXME: is this needed? I mean the caller could still hold the graph node reference...
	shifted := x.Sub(x.Max())
	shifted = shifted.Exp()
	div1 := shifted
	div2 := shifted.Exp().ReduceSum(len(shifted.Shape()) - 1)
	res := div1.Div(div2)
	return res
}

type SoftmaxLayer struct{}

func NewSoftmax() SoftmaxLayer {
	return SoftmaxLayer{}
}

func (l SoftmaxLayer) Parameters(_ bool) []flint_old.Parameter {
	return []flint_old.Parameter{}
}

func (l SoftmaxLayer) Forward(x flint_old.Tensor) flint_old.Tensor {
	return Softmax(x)
}

func (l SoftmaxLayer) TrainMode() {}

func (l SoftmaxLayer) EvalMode() {}

func (l SoftmaxLayer) String() string {
	return "Softmax()"
}
