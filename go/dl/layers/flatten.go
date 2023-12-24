package layers

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
)

/*
Flatten layer takes in an n-dimensional flint (n >= 2) and turns it into a 2-dimensional flint.
It is assumed that the first dimension is for the batches, while the second dimension represents the flattened contents.
*/
type Flatten struct {
}

func NewFlatten() Flatten {
	return Flatten{}
}

func (f Flatten) Parameters(_ bool) []flint.Parameter {
	return []flint.Parameter{}
}

func (f Flatten) Forward(x flint.Tensor) flint.Tensor {
	shape := x.Shape()
	if len(shape) < 2 {
		panic("Flatten: input tensors needs to be at least 2-dimensional")
	}
	var res = x
	for i := len(shape) - 1; i > 1; i-- {
		res = res.FlattenDim(i)
	}
	return res
}

func (f Flatten) TrainMode() {
}

func (f Flatten) EvalMode() {
}

func (f Flatten) String() string {
	return "Flatten()"
}
