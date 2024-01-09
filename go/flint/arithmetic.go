package flint

import (
	"errors"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
)

// add

func addImpl(x Tensor, y Tensor) Tensor {
	node, err := wrapper.AddGraphGraph(x.node, y.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, y.err, err),
	}
}

func Add(x Tensor, y Tensor) Tensor {
	return addImpl(x, y)
}

func (x Tensor) Add(y Tensor) Tensor {
	return addImpl(x, y)
}

func (x *Tensor) Add_(y Tensor) {
	*x = addImpl(*x, y)
}

// sub

func subImpl(x Tensor, y Tensor) Tensor {
	node, err := wrapper.SubGraphGraph(x.node, y.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, y.err, err),
	}
}

func Sub(x Tensor, y Tensor) Tensor {
	return subImpl(x, y)
}

func (x Tensor) Sub(y Tensor) Tensor {
	return subImpl(x, y)
}

func (x *Tensor) Sub_(y Tensor) {
	*x = subImpl(*x, y)
}

// mul

func mulImpl(x Tensor, y Tensor) Tensor {
	node, err := wrapper.MulGraphGraph(x.node, y.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, y.err, err),
	}
}

func Mul(x Tensor, y Tensor) Tensor {
	return mulImpl(x, y)
}

func (x Tensor) Mul(y Tensor) Tensor {
	return mulImpl(x, y)
}

func (x *Tensor) Mul_(y Tensor) {
	*x = mulImpl(*x, y)
}

// div

func divImpl(x Tensor, y Tensor) Tensor {
	node, err := wrapper.DivGraphGraph(x.node, y.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, y.err, err),
	}
}

func Div(x Tensor, y Tensor) Tensor {
	return divImpl(x, y)
}

func (x Tensor) Div(y Tensor) Tensor {
	return divImpl(x, y)
}

func (x *Tensor) Div_(y Tensor) {
	*x = divImpl(*x, y)
}

// pow

func powImpl(x Tensor, y Tensor) Tensor {
	node, err := wrapper.PowGraphGraph(x.node, y.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, y.err, err),
	}
}

func Pow(x Tensor, y Tensor) Tensor {
	return powImpl(x, y)
}

func (x Tensor) Pow(y Tensor) Tensor {
	return powImpl(x, y)
}

func (x *Tensor) Pow_(y Tensor) {
	*x = powImpl(*x, y)
}

// sqrt

func sqrtImpl(x Tensor) Tensor {
	node, err := wrapper.Sqrt(x.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Sqrt(x Tensor) Tensor {
	return sqrtImpl(x)
}

func (x Tensor) Sqrt() Tensor {
	return sqrtImpl(x)
}

func (x *Tensor) Sqrt_() {
	*x = sqrtImpl(*x)
}

// exp

func expImpl(x Tensor) Tensor {
	node, err := wrapper.Exp(x.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Exp(x Tensor) Tensor {
	return expImpl(x)
}

func (x Tensor) Exp() Tensor {
	return expImpl(x)
}

func (x *Tensor) Exp_() {
	*x = expImpl(*x)
}
