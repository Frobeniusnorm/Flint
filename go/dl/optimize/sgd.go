package optimize

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dl/layers"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"log"
)

type Sgd struct {
	baseOptimizer
	learningRate float32
}

func NewSgd(params []layers.Tensor, learningRate float32) Sgd {
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

func (sgd *Sgd) Step(loss layers.Tensor) {
	paramsSimple := make([]flint.GraphNode, len(sgd.params))
	for i, p := range sgd.params {
		paramsSimple[i] = p.Node
	}

	grads := flint.CalculateGradients(loss.Node, paramsSimple)
	for _, g := range grads {
		sgd.calculateUpdate(g, loss.Node)
	}
}

func (sgd *Sgd) String() string {
	return fmt.Sprintf("Sgd(learningRate: %f)", sgd.learningRate)
}
