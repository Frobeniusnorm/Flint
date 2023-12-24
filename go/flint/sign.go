package flint

import (
	"errors"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
)

func negImpl(x Tensor) Tensor {
	node, err := wrapper.Neg(x.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Neg(x Tensor) Tensor {
	return negImpl(x)
}

func (x Tensor) Neg() Tensor {
	return negImpl(x)
}

func (x *Tensor) Neg_() {
	*x = negImpl(*x)
}

func signImpl(x Tensor) Tensor {
	node, err := wrapper.Sign(x.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Sign(x Tensor) Tensor {

	return signImpl(x)
}

func (x Tensor) Sign() Tensor {
	return signImpl(x)
}

func (x *Tensor) Sign_() {
	*x = signImpl(*x)
}

func evenImpl(x Tensor) Tensor {
	node, err := wrapper.Even(x.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Even(x Tensor) Tensor {
	return evenImpl(x)
}

func (x Tensor) Even() Tensor {
	return evenImpl(x)
}

func (x *Tensor) Even_() {
	*x = evenImpl(*x)
}
