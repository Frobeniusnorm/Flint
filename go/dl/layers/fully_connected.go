package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint_old"
)

type FullyConnected struct {
	inputSize      uint
	outputSize     uint
	weightsAndBias flint_old.Parameter
}

// n = input size
// m = output_size

func NewFullyConnected(inputSize uint, outputSize uint) FullyConnected {
	weights := flint_old.Random(flint_old.Shape{inputSize, outputSize})
	bias := flint_old.Constant(int32(1), flint_old.Shape{1, outputSize})
	weightsAndBias := flint_old.NewParameter(weights.Concat(bias, 0))

	return FullyConnected{
		inputSize:      inputSize,
		outputSize:     outputSize,
		weightsAndBias: weightsAndBias,
	}
}

func (fc FullyConnected) Forward(x flint_old.Tensor) flint_old.Tensor {
	inputShape := x.Shape()
	inputShape[len(inputShape)-1] = 1
	ones := flint_old.Constant(int32(1), inputShape)
	combined := x.Concat(ones, uint(len(inputShape)-1))
	res := combined.Matmul(flint_old.Tensor(fc.weightsAndBias))
	return res
}

func (fc FullyConnected) TrainMode() {}

func (fc FullyConnected) EvalMode() {}

func (fc FullyConnected) Parameters(_ bool) []flint_old.Parameter {
	return []flint_old.Parameter{fc.weightsAndBias}
}

func (fc FullyConnected) String() string {
	return fmt.Sprintf("FullyConnected(in: %d, out: %d)", fc.inputSize, fc.outputSize)
}
