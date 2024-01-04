package flint

import (
	"errors"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
)

// matmul

func matmulImpl(x Tensor, y Tensor) Tensor {
	node, err := wrapper.Matmul(x.node, y.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, y.err, err),
	}
}

func Matmul(x Tensor, y Tensor) Tensor {
	return matmulImpl(x, y)
}

func (x Tensor) Matmul(y Tensor) Tensor {
	return matmulImpl(x, y)
}

func (x *Tensor) Matmul_(y Tensor) {
	*x = matmulImpl(*x, y)
}

// convert

func convertImpl(x Tensor, dtype wrapper.DataType) Tensor {
	node, err := wrapper.Convert(x.node, dtype)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Convert(x Tensor, dtype wrapper.DataType) Tensor {
	return convertImpl(x, dtype)
}

func (x Tensor) Convert(dtype wrapper.DataType) Tensor {
	return convertImpl(x, dtype)
}

func (x *Tensor) Convert_(dtype wrapper.DataType) {
	*x = convertImpl(*x, dtype)
}
