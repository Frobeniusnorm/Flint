package layers

type Flatten struct {
	BaseLayer
}

func NewFlatten() Flatten {
	return Flatten{}
}

func (fl Flatten) Parameters(recurse bool) []Parameter {
	return []Parameter{}
}

func (fl Flatten) Forward(x Tensor) Tensor {
	return x // TODO
}

func (fl Flatten) Train() {}

func (fl Flatten) Eval() {}

func (fl Flatten) String() string {
	return "Flatten()"
}
