package optimize

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dl/layers"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type baseOptimizer struct {
	params      []layers.Tensor
	regularizer Regularizer
}

type baseRegularizer struct {
}

type Regularizer interface{}

type Optimizer interface {
	fmt.Stringer
	calculateUpdate(weightTensor flint.GraphNode, gradientTensor flint.GraphNode) flint.GraphNode
	Step()
}
