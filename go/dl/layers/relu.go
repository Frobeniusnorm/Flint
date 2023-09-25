package layers

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
		input: nil,
	}
}

func (relu *ReLU) Forward(x Tensor) Tensor {
	return x
}

func (relu *ReLU) Backward(x Tensor) Tensor {
	return x
}
