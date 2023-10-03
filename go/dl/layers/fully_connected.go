package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type FullyConnected struct {
	inputSize      uint
	outputSize     uint
	weightsAndBias Parameter
}

// n = input size
// m = output_size

func NewFullyConnected(inputSize uint, outputSize uint) FullyConnected {
	weights := flint.CreateGraphRandom(flint.Shape{inputSize, outputSize})
	bias := flint.CreateGraphConstant(1, flint.Shape{1, outputSize}, flint.F_FLOAT32)
	weightsAndBias := NewParameter(flint.Concat(weights, bias, 0))

	return FullyConnected{
		inputSize:      inputSize,
		outputSize:     outputSize,
		weightsAndBias: weightsAndBias,
	}
}

func (fc FullyConnected) Forward(x Tensor) Tensor {
	inputShape := x.Node.GetShape()
	inputShape[len(inputShape)-1] = 1
	ones := flint.CreateGraphConstant(1, inputShape, flint.F_INT32)
	combined := flint.Concat(x.Node, ones, uint(len(inputShape)-1))
	res := flint.Matmul(combined, fc.weightsAndBias.Node)
	return NewTensor(res)
}

func (fc FullyConnected) Train() {}

func (fc FullyConnected) Eval() {}

func (fc FullyConnected) Parameters(recurse bool) []Parameter {
	return []Parameter{fc.weightsAndBias}
}

func (fc FullyConnected) String() string {
	return fmt.Sprintf("FullyConnected(in: %d, out: %d)", fc.inputSize, fc.outputSize)
}
