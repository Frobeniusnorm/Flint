package layers

type Flatten struct {
	BaseLayer
}

func NewFlatten() Flatten {
	return Flatten{
		BaseLayer: BaseLayer{
			trainable:  false,
			EnableGrad: false,
		}}
}

func (fl Flatten) Forward(x Tensor) Tensor {
	return x // TODO
}

func (fl Flatten) String() string {
	return "Flatten()"
}
