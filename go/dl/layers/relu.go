package layers

import "github.com/Frobeniusnorm/Flint/go/flint"

type ReLU struct {
	BaseLayer
	input Tensor
}

func NewRelu() ReLU {
	return ReLU{
		BaseLayer: BaseLayer{
			trainable:     false,
			trainingPhase: false,
		},
		input: Tensor{},
	}
}

func (relu *ReLU) Forward(x Tensor) Tensor {
	res := flint.Max(x.node, 0)
	return Tensor{node: res}
}

func (relu *ReLU) Backward(x Tensor) Tensor {
	return x
}

func (relu *ReLU) String() string {
	return "ReLU()"
}
