package layers

import "github.com/Frobeniusnorm/Flint/go/flint"

type ReLU struct {
	BaseLayer
}

func NewRelu() ReLU {
	return ReLU{
		BaseLayer: BaseLayer{
			trainable:  false,
			EnableGrad: false,
		},
	}
}

func (relu *ReLU) Forward(x Tensor) Tensor {
	res := flint.Max(x.node, int32(0))
	return Tensor{node: res}
}

func (relu *ReLU) Backward(x Tensor) Tensor {
	return x
}

func (relu *ReLU) String() string {
	return "ReLU()"
}
