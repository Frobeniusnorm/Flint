package optimize

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/Frobeniusnorm/Flint/go/tensor"
	"log"
)

type Sgd struct {
	baseOptimizer
	learningRate float32
}

func NewSgd(params []tensor.Parameter, learningRate float32) Sgd {
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

func (sgd *Sgd) Step(loss tensor.Tensor) {
	// turn tensor array into array of graph nodes
	paramsSimple := make([]flint.GraphNode, len(sgd.params))
	for i, p := range sgd.params {
		paramsSimple[i] = p.node
	}

	grads := flint.CalculateGradients(loss.node, paramsSimple)
	for i, w := range paramsSimple {
		for _, g := range grads {
			sgd.params[i].Close()
			update := sgd.calculateUpdate(g, w)
			sgd.params[i] = tensor.NewParameter(update)
		}
	}
}

func (sgd *Sgd) String() string {
	return fmt.Sprintf("Sgd(learningRate: %f, params: %d)", sgd.learningRate, len(sgd.params))
}
