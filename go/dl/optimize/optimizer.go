package optimize

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/tensor"
)

type baseOptimizer struct {
	params      []tensor.Parameter
	regularizer Regularizer
}

type Optimizer interface {
	fmt.Stringer
	calculateUpdate(weightTensor tensor.Tensor, gradientTensor tensor.Tensor) tensor.Tensor
	Step(loss tensor.Tensor)
}
