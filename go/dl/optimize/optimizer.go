package optimize

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/Frobeniusnorm/Flint/go/tensor"
)

type baseOptimizer struct {
	params      []tensor.Parameter
	regularizer Regularizer
}

type Optimizer interface {
	fmt.Stringer
	calculateUpdate(weightTensor flint.GraphNode, gradientTensor flint.GraphNode) flint.GraphNode
	Step(loss tensor.Tensor)
}
