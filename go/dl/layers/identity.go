package layers

type Identity struct {
	BaseLayer
}

func NewIdentity() Identity {
	return Identity{BaseLayer{
		trainable:  false,
		EnableGrad: false,
	}}
}

func (id *Identity) Forward(x Tensor) Tensor {
	return x
}

func (id *Identity) String() string {
	return "Identity()"
}
