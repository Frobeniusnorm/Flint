package layers

import (
	"fmt"
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
	weights := tensor.Random(tensor.Shape{inputSize, outputSize})
	bias := tensor.Constant(int32(1), tensor.Shape{1, outputSize})
	weightsAndBias := tensor.NewParameter(tensor.Concat(weights, bias, 0))

	return FullyConnected{
		inputSize:      inputSize,
		outputSize:     outputSize,
		weightsAndBias: weightsAndBias,
	}
}

func (fc FullyConnected) Forward(x tensor.Tensor) tensor.Tensor {
	inputShape := x.Shape()
	inputShape[len(inputShape)-1] = 1
	ones := tensor.Constant(int32(1), inputShape)
	combined := tensor.Concat(x, ones, uint(len(inputShape)-1))
	res := combined.Matmul(fc.weightsAndBias)
	return res
}

func (fc FullyConnected) TrainMode() {}

func (fc FullyConnected) EvalMode() {}

func (fc FullyConnected) Parameters(_ bool) []tensor.Parameter {
	return []tensor.Parameter{fc.weightsAndBias}
}

func (fc FullyConnected) String() string {
	return fmt.Sprintf("FullyConnected(in: %d, out: %d)", fc.inputSize, fc.outputSize)
}
