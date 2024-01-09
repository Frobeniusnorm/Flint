package flint

import (
	"errors"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
)

// cos

func cosImpl(x Tensor) Tensor {
	node, err := wrapper.Cos(x.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Cos(x Tensor) Tensor {
	return cosImpl(x)
}

func (x Tensor) Cos() Tensor {
	return cosImpl(x)
}

func (x *Tensor) Cos_() {
	*x = cosImpl(*x)
}

// sin

func sinImpl(x Tensor) Tensor {
	node, err := wrapper.Sin(x.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Sin(x Tensor) Tensor {
	return sinImpl(x)
}

func (x Tensor) Sin() Tensor {
	return sinImpl(x)
}

func (x *Tensor) Sin_() {
	*x = sinImpl(*x)
}

// tan

func tanImpl(x Tensor) Tensor {
	node, err := wrapper.Tan(x.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Tan(x Tensor) Tensor {
	return tanImpl(x)
}

func (x Tensor) Tan() Tensor {
	return tanImpl(x)
}

func (x *Tensor) Tan_() {
	*x = tanImpl(*x)
}

// acos

func acosImpl(x Tensor) Tensor {
	node, err := wrapper.Acos(x.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Acos(x Tensor) Tensor {
	return acosImpl(x)
}

func (x Tensor) Acos() Tensor {
	return acosImpl(x)
}

func (x *Tensor) Acos_() {
	*x = acosImpl(*x)
}

// asin

func asinImpl(x Tensor) Tensor {
	node, err := wrapper.Asin(x.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Asin(x Tensor) Tensor {
	return asinImpl(x)
}

func (x Tensor) Asin() Tensor {
	return asinImpl(x)
}

func (x *Tensor) Asin_() {

	*x = asinImpl(*x)
}

// atan

func atanImpl(x Tensor) Tensor {
	node, err := wrapper.Atan(x.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Atan(x Tensor) Tensor {
	return atanImpl(x)
}

func (x Tensor) Atan() Tensor {
	return atanImpl(x)
}

func (x *Tensor) Atan_() {
	*x = atanImpl(*x)
}
