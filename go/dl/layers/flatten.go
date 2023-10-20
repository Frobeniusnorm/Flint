package layers

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/Frobeniusnorm/Flint/go/tensor"
)

/*
Flatten layer takes in an n-dimensional tensor (n >= 2) and turns it into a 2-dimensional tensor.
It is assumed that the first dimension is for the batches, while the second dimension represents the flattened contents.
*/
type Flatten struct {
}

func NewFlatten() Flatten {
	return Flatten{}
}

func (f Flatten) Parameters(_ bool) []tensor.Parameter {
	return []tensor.Parameter{}
}

func (f Flatten) Forward(x tensor.Tensor) tensor.Tensor {
	shape := x.Node.GetShape()
	if len(shape) < 2 {
		panic("Flatten: input tensors needs to be at least 2-dimensional")
	}
	var res = x.Node
	for i := len(shape) - 1; i > 1; i-- {
		res = flint.FlattenDim(res, i)
	}
	return tensor.NewTensor(res)
}

func (f Flatten) TrainMode() {
}

func (f Flatten) EvalMode() {
}

func (f Flatten) String() string {
	return "Flatten()"
}
