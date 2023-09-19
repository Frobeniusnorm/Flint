package layers

type Tensor struct {
}

type Layer struct {
	trainable bool
}

type BaseLayer interface {
	forward(x Tensor) Tensor
}

type Dropout struct {
	Layer       // embed base layer
	probability float32
	_output     Tensor
}
