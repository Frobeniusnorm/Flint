package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/Frobeniusnorm/Flint/go/tensor"
)

type FullyConnected struct {
	inputSize      uint
	outputSize     uint
	weightsAndBias tensor.Parameter
}

// n = input size
// m = output_size

func NewFullyConnected(inputSize uint, outputSize uint) FullyConnected {
	weights := flint.CreateGraphRandom(flint.Shape{inputSize, outputSize})
	bias := flint.CreateGraphConstant(1, flint.Shape{1, outputSize})
	weightsAndBias := tensor.NewParameter(flint.Concat(weights, bias, 0))

	return FullyConnected{
		inputSize:      inputSize,
		outputSize:     outputSize,
		weightsAndBias: weightsAndBias,
	}
}

func (fc FullyConnected) Forward(x tensor.Tensor) tensor.Tensor {
	inputShape := x.node.GetShape()
	inputShape[len(inputShape)-1] = 1
	ones := flint.CreateGraphConstant(1, inputShape)
	combined := flint.Concat(x.node, ones, uint(len(inputShape)-1))
	res := flint.Matmul(combined, fc.weightsAndBias.node)
	return tensor.NewTensor(res)
}

func (fc FullyConnected) TrainMode() {}

func (fc FullyConnected) EvalMode() {}

func (fc FullyConnected) Parameters(_ bool) []tensor.Parameter {
	return []tensor.Parameter{fc.weightsAndBias}
}

func (fc FullyConnected) String() string {
	return fmt.Sprintf("FullyConnected(in: %d, out: %d)", fc.inputSize, fc.outputSize)
}
