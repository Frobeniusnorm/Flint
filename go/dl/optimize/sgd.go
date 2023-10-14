package optimize

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dl"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"log"
)

type Sgd struct {
	baseOptimizer
	learningRate float32
}

func NewSgd(params []dl.Parameter, learningRate float32) Sgd {
	if learningRate < 0.0 {
		log.Panicf("Invalid learning rate: %f", learningRate)
	}
	return Sgd{
		baseOptimizer: baseOptimizer{
			params:      params,
			regularizer: nil,
		},
		learningRate: learningRate,
	}
}

func (sgd *Sgd) calculateUpdate(weightTensor flint.GraphNode, gradientTensor flint.GraphNode) flint.GraphNode {
	return flint.Sub(weightTensor, flint.Mul(gradientTensor, sgd.learningRate))
}

func (sgd *Sgd) Step(loss dl.Tensor) {
	// turn tensor array into array of graph nodes
	paramsSimple := make([]flint.GraphNode, len(sgd.params))
	for i, p := range sgd.params {
		paramsSimple[i] = p.Node
	}

	grads := flint.CalculateGradients(loss.Node, paramsSimple)
	for i, w := range paramsSimple {
		for _, g := range grads { // FIXME: it's all broken
			sgd.params[i].Node = sgd.calculateUpdate(g, w)
		}

	}
}

func (sgd *Sgd) String() string {
	return fmt.Sprintf("Sgd(learningRate: %f, params: %d)", sgd.learningRate, len(sgd.params))
}
