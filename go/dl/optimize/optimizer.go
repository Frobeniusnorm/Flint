package optimize

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type baseOptimizer struct {
	params      []flint.Parameter
	regularizer Regularizer
}

type Optimizer interface {
	fmt.Stringer
	calculateUpdate(weightTensor flint.Tensor, gradientTensor flint.Tensor) flint.Tensor
	Step(loss flint.Tensor)
}
