package optim

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dl"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type baseOptimizer struct {
	params      []dl.Parameter
	regularizer Regularizer
}

type Optimizer interface {
	fmt.Stringer
	calculateUpdate(weightTensor flint.GraphNode, gradientTensor flint.GraphNode) flint.GraphNode
	Step(loss dl.Tensor)
}
