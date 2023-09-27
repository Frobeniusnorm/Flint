package optimize

import "github.com/Frobeniusnorm/Flint/go/flint"

type Sgd struct {
	baseOptimizer
	learningRate float32
}

func NewSgd(learningRate float32) Sgd {
	return Sgd{
		baseOptimizer: baseOptimizer{regularizer: nil},
		learningRate:  learningRate,
	}
}

func (sgd *Sgd) Update(weightTensor flint.GraphNode, gradientTensor flint.GraphNode) flint.GraphNode {
	return flint.Sub(weightTensor, flint.Mul(gradientTensor, sgd.learningRate))
}
