package layers

import "fmt"

type Dropout struct {
	BaseLayer   // embed base layer
	probability float32
	output      Tensor
}

func NewDropout(prob float32) Dropout {
	return Dropout{
		BaseLayer: BaseLayer{
			trainable:  false,
			EnableGrad: false,
		},
		probability: prob,
		output:      Tensor{},
	}
}

func (d Dropout) Forward(x Tensor) Tensor {
	return x
}

func (d Dropout) Backward(x Tensor) Tensor {
	return x
}

func (d Dropout) String() string {
	return fmt.Sprintf("Dropout(prob=%f)", d.probability)
}
