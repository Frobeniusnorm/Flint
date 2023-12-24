package optimize

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
	"log"
)

type Sgd struct {
	baseOptimizer
	learningRate float32
}

func NewSgd(params []flint.Parameter, learningRate float32) Sgd {
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

func (sgd *Sgd) calculateUpdate(weightTensor flint.Tensor, gradientTensor flint.Tensor) flint.Tensor {
	return weightTensor.Sub(gradientTensor.Mul(flint.Scalar(sgd.learningRate)))
}

func (sgd *Sgd) Step(loss flint.Tensor) {
	// turn flint array into array of graph nodes
	paramsSimple := make([]wrapper.GraphNode, len(sgd.params))
	for i, p := range sgd.params {
		paramsSimple[i] = p.Node()
	}

	grads, _ := wrapper.CalculateGradients(loss.Node(), paramsSimple)
	for i, w := range sgd.params {
		for _, g := range grads {
			sgd.params[i].Close()
			gN := flint.FromNode(g)
			update := sgd.calculateUpdate(gN, flint.Tensor(w))
			gN.Close()
			sgd.params[i] = flint.NewParameter(update)
		}
	}
}

func (sgd *Sgd) String() string {
	return fmt.Sprintf("Sgd(learningRate: %f, params: %d)", sgd.learningRate, len(sgd.params))
}
