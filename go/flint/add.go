package flint

import (
	"errors"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
)

type Tensor struct {
	node *wrapper.GraphNode
	err  error
	//data        *any // only in case the actual type doesn't matter. Else use one of the below
	dataInt32   *int32
	dataInt64   *int64
	dataFloat32 *float32
	dataFloat64 *float64
	dataType    wrapper.DataType
}

func AddImpl(x Tensor, y Tensor) Tensor {
	node, err := wrapper.AddGraphFloat(x.node, y.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, y.err, err),
	}
}

func Add(x Tensor, y Tensor) Tensor {
	return AddImpl(x, y)
}

func (x Tensor) Add(y Tensor) Tensor {
	return AddImpl(x, y)
}

func (x *Tensor) Add_(y Tensor) {
	*x = AddImpl(*x, y)
}
