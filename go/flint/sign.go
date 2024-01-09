package flint

import (
	"errors"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
)

// neg

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

// sign

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

// even

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

// less

func lessImpl(x Tensor, y Tensor) Tensor {
	node, err := wrapper.LessGraphGraph(x.node, y.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, y.err, err),
	}
}

func Less(x Tensor, y Tensor) Tensor {
	return lessImpl(x, y)
}

func (x Tensor) Less(y Tensor) Tensor {
	return lessImpl(x, y)
}

func (x *Tensor) Less_(y Tensor) {
	*x = lessImpl(*x, y)
}

// greater

func greaterImpl(x Tensor, y Tensor) Tensor {
	node, err := wrapper.GreaterGraphGraph(x.node, y.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, y.err, err),
	}
}

func Greater(x Tensor, y Tensor) Tensor {
	return greaterImpl(x, y)
}

func (x Tensor) Greater(y Tensor) Tensor {
	return greaterImpl(x, y)
}

func (x *Tensor) Greater_(y Tensor) {
	*x = greaterImpl(*x, y)
}

// equal

func equalImpl(x Tensor, y Tensor) Tensor {
	node, err := wrapper.EqualGraphGraph(x.node, y.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, y.err, err),
	}
}

func Equal(x Tensor, y Tensor) Tensor {
	return equalImpl(x, y)
}

func (x Tensor) Equal(y Tensor) Tensor {
	return equalImpl(x, y)
}

func (x *Tensor) Equal_(y Tensor) {
	*x = equalImpl(*x, y)
}

// abs

func absImpl(x Tensor) Tensor {
	node, err := wrapper.Abs(x.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Abs(x Tensor) Tensor {
	return absImpl(x)
}

func (x Tensor) Abs() Tensor {
	return absImpl(x)
}

func (x *Tensor) Abs_() {
	*x = absImpl(*x)
}
