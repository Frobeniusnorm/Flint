package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type FullyConnected struct {
	BaseLayer
	inputSize  uint
	outputSize uint
	weights    Tensor //Parameter
	bias       Tensor //Parameter
}

func NewFullyConnected(inputSize uint, outputSize uint) FullyConnected {
	weights := NewTensor(flint.CreateGraphRandom(flint.Shape{1 + inputSize, outputSize}))
	bias := NewTensor(flint.CreateGraphConstant(1, flint.Shape{outputSize}, flint.F_FLOAT32))
	return FullyConnected{
		BaseLayer: BaseLayer{
			trainable:  true,
			EnableGrad: true,
		},
		inputSize:  inputSize,
		outputSize: outputSize,
		weights:    weights,
		bias:       bias,
	}
}

func (fc *FullyConnected) Forward(x Tensor) Tensor {
	ones := flint.CreateGraphConstant(1, flint.Shape{x.node.GetShape()[0], 1}, flint.F_INT32)
	combined := flint.Concat(x.node, ones, 1)
	res := flint.Matmul(combined, fc.weights.node)
	return NewTensor(res)
}

func (fc *FullyConnected) String() string {
	return fmt.Sprintf("FullyConnected(in: %d, out: %d)", fc.inputSize, fc.outputSize)
}
