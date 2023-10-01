package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type FullyConnected struct {
	BaseLayer
	inputSize      uint
	outputSize     uint
	weightsAndBias Tensor
}

// n = input size
// m = output_size

func NewFullyConnected(inputSize uint, outputSize uint) FullyConnected {
	weights := flint.CreateGraphRandom(flint.Shape{inputSize, outputSize})
	bias := flint.CreateGraphConstant(1, flint.Shape{1, outputSize}, flint.F_FLOAT32)
	weightsAndBias := NewTensor(flint.Concat(weights, bias, 0))

	return FullyConnected{
		BaseLayer: BaseLayer{
			trainable:  true,
			EnableGrad: true,
		},
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

	fmt.Println("weights + bias shape:", fc.weightsAndBias.Node.GetShape())
	fmt.Println("ones shape:", ones.GetShape())
	fmt.Println("input shape:", x.Node.GetShape())
	fmt.Println("combined shape:", combined.GetShape())
	res := flint.Matmul(combined, fc.weightsAndBias.Node)
	return NewTensor(res)
}

func (fc FullyConnected) Parameters(recurse bool) []Tensor {
	return []Tensor{fc.weightsAndBias}
}

func (fc FullyConnected) String() string {
	return fmt.Sprintf("FullyConnected(in: %d, out: %d)", fc.inputSize, fc.outputSize)
}
