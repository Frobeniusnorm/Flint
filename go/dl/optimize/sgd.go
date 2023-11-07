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

func (sgd *Sgd) calculateUpdate(weightTensor tensor.Tensor, gradientTensor tensor.Tensor) tensor.Tensor {
	return weightTensor.Sub(gradientTensor.Mul(tensor.Scalar(sgd.learningRate)))
}

func (sgd *Sgd) Step(loss tensor.Tensor) {
	// turn tensor array into array of graph nodes
	paramsSimple := make([]flint.GraphNode, len(sgd.params))
	for i, p := range sgd.params {
		paramsSimple[i] = p.Node()
	}

	grads, _ := flint.CalculateGradients(loss.Node(), paramsSimple)
	for i, w := range sgd.params {
		for _, g := range grads {
			sgd.params[i].Close()
			gN := tensor.FromNode(g)
			update := sgd.calculateUpdate(gN, tensor.Tensor(w))
			gN.Close()
			sgd.params[i] = tensor.NewParameter(update)
		}
	}
}

func (sgd *Sgd) String() string {
	return fmt.Sprintf("Sgd(learningRate: %f, params: %d)", sgd.learningRate, len(sgd.params))
}
