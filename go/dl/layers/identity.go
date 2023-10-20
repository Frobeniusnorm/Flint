package layers

import (
	"github.com/Frobeniusnorm/Flint/go/tensor"
)

/*
Identity is a layer that essentially does nothing.
The tensor is just passed through without any changes and the gradient is constant as well.

An example usage might be the construction of a network with a few given conditions.
If the condition is not fulfilled you might add an [Identity] layer instead of just leaving it out.
*/
type Identity struct{}

func NewIdentity() Identity {
	return Identity{}
}

func (id Identity) Parameters(_ bool) []tensor.Parameter {
	return []tensor.Parameter{}
}

func (id Identity) Forward(x tensor.Tensor) tensor.Tensor {
	return x
}

func (id Identity) TrainMode() {}

func (id Identity) EvalMode() {}

func (id Identity) String() string {
	return "Identity()"
}
