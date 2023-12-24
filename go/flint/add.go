package flint

import (
	"errors"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
)

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
