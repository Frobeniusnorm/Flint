package layers

type Identity struct {
	BaseLayer
}

func NewIdentity() Identity {
	return Identity{}
}

func (id Identity) Parameters(recurse bool) []Parameter {
	return []Parameter{}
}

func (id Identity) Forward(x Tensor) Tensor {
	return x
}

func (id Identity) Train() {}

func (id Identity) Eval() {}

func (id Identity) String() string {
	return "Identity()"
}
