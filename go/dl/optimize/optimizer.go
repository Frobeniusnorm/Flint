package optimize

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint_old"
)

type baseOptimizer struct {
	params      []flint_old.Parameter
	regularizer Regularizer
}

type Optimizer interface {
	fmt.Stringer
	calculateUpdate(weightTensor flint_old.Tensor, gradientTensor flint_old.Tensor) flint_old.Tensor
	Step(loss flint_old.Tensor)
}
