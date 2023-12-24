package flint

import "github.com/Frobeniusnorm/Flint/go/wrapper"

type Tensor struct {
	node wrapper.GraphNode
	err  error
}

func Add(x Tensor, y Tensor) Tensor {
	node, err := wrapper.Add(x.node, y.node)
	return Tensor{
		node: node,
		err:  err,
	}
}

func (x Tensor) Add(y Tensor) Tensor {
	return Add(x, y)
}
