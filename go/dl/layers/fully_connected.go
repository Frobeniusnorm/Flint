package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type FullyConnected struct {
	inputSize      uint
	outputSize     uint
	weightsAndBias flint.Parameter
}

// n = input size
// m = output_size

func NewFullyConnected(inputSize uint, outputSize uint) FullyConnected {
	weights := flint.Random(flint.Shape{inputSize, outputSize})
	bias := flint.Constant(int32(1), flint.Shape{1, outputSize})
	weightsAndBias := flint.NewParameter(weights.Concat(bias, 0))

	return FullyConnected{
		inputSize:      inputSize,
		outputSize:     outputSize,
		weightsAndBias: weightsAndBias,
	}
}

func (fc FullyConnected) Forward(x flint.Tensor) flint.Tensor {
	inputShape := x.Shape()
	inputShape[len(inputShape)-1] = 1
	ones := flint.Constant(int32(1), inputShape)
	combined := x.Concat(ones, uint(len(inputShape)-1))
	res := combined.Matmul(flint.Tensor(fc.weightsAndBias))
	return res
}

func (fc FullyConnected) TrainMode() {}

func (fc FullyConnected) EvalMode() {}

func (fc FullyConnected) Parameters(_ bool) []flint.Parameter {
	return []flint.Parameter{fc.weightsAndBias}
}

func (fc FullyConnected) String() string {
	return fmt.Sprintf("FullyConnected(in: %d, out: %d)", fc.inputSize, fc.outputSize)
}
