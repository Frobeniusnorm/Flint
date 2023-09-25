package layers

type Dropout struct {
	BaseLayer   // embed base layer
	probability float32
	output      Tensor
}

func NewDropout(prob float32) Dropout {
	return Dropout{
		BaseLayer: BaseLayer{
			trainable:     false,
			trainingPhase: false,
		},
		probability: prob,
		output:      nil,
	}
}

func (d *Dropout) Forward(x Tensor) Tensor {
	return x
}

func (d *Dropout) Backward(x Tensor) Tensor {
	return x
}
