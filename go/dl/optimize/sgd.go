package optimize

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint_old"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
	"log"
)

type Sgd struct {
	baseOptimizer
	learningRate float32
}

func NewSgd(params []flint_old.Parameter, learningRate float32) Sgd {
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

func (sgd *Sgd) calculateUpdate(weightTensor flint_old.Tensor, gradientTensor flint_old.Tensor) flint_old.Tensor {
	return weightTensor.Sub(gradientTensor.Mul(flint_old.Scalar(sgd.learningRate)))
}

func (sgd *Sgd) Step(loss flint_old.Tensor) {
	// turn flint_old array into array of graph nodes
	paramsSimple := make([]wrapper.GraphNode, len(sgd.params))
	for i, p := range sgd.params {
		paramsSimple[i] = p.Node()
	}

	grads, _ := wrapper.CalculateGradients(loss.Node(), paramsSimple)
	for i, w := range sgd.params {
		for _, g := range grads {
			sgd.params[i].Close()
			gN := flint_old.FromNode(g)
			update := sgd.calculateUpdate(gN, flint_old.Tensor(w))
			gN.Close()
			sgd.params[i] = flint_old.NewParameter(update)
		}
	}
}

func (sgd *Sgd) String() string {
	return fmt.Sprintf("Sgd(learningRate: %f, params: %d)", sgd.learningRate, len(sgd.params))
}
