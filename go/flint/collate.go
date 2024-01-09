package flint

import (
	"errors"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
)

// minimum

func minimumImpl(x Tensor, y Tensor) Tensor {
	node, err := wrapper.MinimumGraphGraph(x.node, y.node)
	return Tensor{
		node: node,
		err:  errors.Join(err, x.err, y.err),
	}
}

func Minimum(x Tensor, y Tensor) Tensor {
	return minimumImpl(x, y)
}

func (x Tensor) Minimum(y Tensor) Tensor {
	return minimumImpl(x, y)
}

func (x *Tensor) Minimum_(y Tensor) {
	*x = minimumImpl(*x, y)
}

// maximum

func maximumImpl(x Tensor, y Tensor) Tensor {
	node, err := wrapper.MaximumGraphGraph(x.node, y.node)
	return Tensor{
		node: node,
		err:  errors.Join(err, x.err, y.err),
	}
}

func Maximum(x Tensor, y Tensor) Tensor {
	return maximumImpl(x, y)
}

func (x Tensor) Maximum(y Tensor) Tensor {
	return maximumImpl(x, y)
}

func (x *Tensor) Maximum_(y Tensor) {
	*x = maximumImpl(*x, y)
}

// reduce sum

func reduceSumImpl(x Tensor, dim int) Tensor {
	node, err := wrapper.ReduceSum(x.node, dim)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func ReduceSum(x Tensor, dim int) Tensor {
	return reduceSumImpl(x, dim)
}

func (x Tensor) ReduceSum(dim int) Tensor {
	return reduceSumImpl(x, dim)
}

func (x *Tensor) ReduceSum_(dim int) {
	*x = reduceSumImpl(*x, dim)
}

// reduce mul

func reduceMulImpl(x Tensor, dim int) Tensor {
	node, err := wrapper.ReduceMul(x.node, dim)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func ReduceMul(x Tensor, dim int) Tensor {
	return reduceMulImpl(x, dim)
}

func (x Tensor) ReduceMul(dim int) Tensor {
	return reduceMulImpl(x, dim)
}

func (x *Tensor) ReduceMul_(dim int) {
	*x = reduceMulImpl(*x, dim)
}

// reduce min

func reduceMinImpl(x Tensor, dim int) Tensor {
	node, err := wrapper.ReduceMin(x.node, dim)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func ReduceMin(x Tensor, dim int) Tensor {
	return reduceMinImpl(x, dim)
}

func (x Tensor) ReduceMin(dim int) Tensor {
	return reduceMinImpl(x, dim)
}

func (x *Tensor) ReduceMin_(dim int) {
	*x = reduceMinImpl(*x, dim)
}

// reduce max

func reduceMaxImpl(x Tensor, dim int) Tensor {
	node, err := wrapper.ReduceMax(x.node, dim)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func ReduceMax(x Tensor, dim int) Tensor {
	return reduceMaxImpl(x, dim)
}

func (x Tensor) ReduceMax(dim int) Tensor {
	return reduceMaxImpl(x, dim)
}

func (x *Tensor) ReduceMax_(dim int) {
	*x = reduceMaxImpl(*x, dim)
}

// flatten

func flattenImpl(x Tensor) Tensor {
	node, err := wrapper.Flatten(x.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Flatten(x Tensor) Tensor {
	return flattenImpl(x)
}

func (x Tensor) Flatten() Tensor {
	return flattenImpl(x)
}

func (x *Tensor) Flatten_() {
	*x = flattenImpl(*x)
}

// flatten dim

func flattenDimImpl(x Tensor, dim int) Tensor {
	node, err := wrapper.FlattenDim(x.node, dim)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func FlattenDim(x Tensor, dim int) Tensor {
	return flattenDimImpl(x, dim)
}

func (x Tensor) FlattenDim(dim int) Tensor {
	return flattenDimImpl(x, dim)
}

func (x *Tensor) FlattenDim_(dim int) {
	*x = flattenDimImpl(*x, dim)
}

// pooling sum

func poolingSumImpl(x Tensor, size Shape, stride Stride) Tensor {
	node, err := wrapper.PoolingSum(x.node, size, stride)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func PoolingSum(x Tensor, size Shape, stride Stride) Tensor {
	return poolingSumImpl(x, size, stride)
}

func (x Tensor) PoolingSum(size Shape, stride Stride) Tensor {
	return poolingSumImpl(x, size, stride)
}

func (x *Tensor) PoolingSum_(size Shape, stride Stride) {
	*x = poolingSumImpl(*x, size, stride)
}

// pooling max

func poolingMaxImpl(x Tensor, size Shape, stride Stride) Tensor {
	node, err := wrapper.PoolingMax(x.node, size, stride)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func PoolingMax(x Tensor, size Shape, stride Stride) Tensor {
	return poolingMaxImpl(x, size, stride)
}

func (x Tensor) PoolingMax(size Shape, stride Stride) Tensor {
	return poolingMaxImpl(x, size, stride)
}

func (x *Tensor) PoolingMax_(size Shape, stride Stride) {
	*x = poolingMaxImpl(*x, size, stride)
}
