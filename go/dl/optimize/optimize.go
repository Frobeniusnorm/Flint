package optimize

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type baseOptimizer struct {
	regularizer Regularizer
}

type baseRegularizer struct {
}

type Regularizer interface{}

type Optimizer interface {
	Update(weightTensor flint.GraphNode, gradientTensor flint.GraphNode) flint.GraphNode
}
