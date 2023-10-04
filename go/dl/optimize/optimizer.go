package optimize

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dl/layers"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type baseOptimizer struct {
	params      []layers.Parameter
	regularizer Regularizer
}

type Optimizer interface {
	fmt.Stringer
	calculateUpdate(weightTensor flint.GraphNode, gradientTensor flint.GraphNode) flint.GraphNode
	Step(loss layers.Tensor)
}
