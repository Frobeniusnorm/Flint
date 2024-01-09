package flint

import (
	"errors"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
)

// reshape

func reshapeImpl(x Tensor, shape Shape) Tensor {
	node, err := wrapper.Reshape(x.node, shape)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Reshape(x Tensor, shape Shape) Tensor {
	return reshapeImpl(x, shape)
}

func (x Tensor) Reshape(shape Shape) Tensor {
	return reshapeImpl(x, shape)
}

func (x *Tensor) Reshape_(shape Shape) {
	*x = reshapeImpl(*x, shape)
}

// slice

func sliceImpl(x Tensor, start Axes, end Axes) Tensor {
	node, err := wrapper.Slice(x.node, start, end)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Slice(x Tensor, start Axes, end Axes) Tensor {
	return sliceImpl(x, start, end)
}

func (x Tensor) Slice(start Axes, end Axes) Tensor {
	return sliceImpl(x, start, end)
}

func (x *Tensor) Slice_(start Axes, end Axes) {
	*x = sliceImpl(*x, start, end)
}

// slice with stride

func sliceWithStrideImpl(x Tensor, start Axes, end Axes, stride Stride) Tensor {
	node, err := wrapper.SliceWithStride(x.node, start, end, stride)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func SliceWithStride(x Tensor, start Axes, end Axes, stride Stride) Tensor {
	return sliceWithStrideImpl(x, start, end, stride)
}

func (x Tensor) SliceWithStride(start Axes, end Axes, stride Stride) Tensor {
	return sliceWithStrideImpl(x, start, end, stride)
}

func (x *Tensor) SliceWithStride_(start Axes, end Axes, stride Stride) {
	*x = sliceWithStrideImpl(*x, start, end, stride)
}

// extend

func extendImpl(x Tensor, shape Shape, insertAt Axes) Tensor {
	node, err := wrapper.Extend(x.node, shape, insertAt)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Extend(x Tensor, shape Shape, insertAt Axes) Tensor {
	return extendImpl(x, shape, insertAt)
}

func (x Tensor) Extend(shape Shape, insertAt Axes) Tensor {
	return extendImpl(x, shape, insertAt)
}

func (x *Tensor) Extend_(shape Shape, insertAt Axes) {
	*x = extendImpl(*x, shape, insertAt)
}

// extend with stride

func extendWithStrideImpl(x Tensor, shape Shape, insertAt Axes, stride Stride) Tensor {
	node, err := wrapper.ExtendWithStride(x.node, shape, insertAt, stride)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func ExtendWithStride(x Tensor, shape Shape, insertAt Axes, stride Stride) Tensor {
	return extendWithStrideImpl(x, shape, insertAt, stride)
}

func (x Tensor) ExtendWithStride(shape Shape, insertAt Axes, stride Stride) Tensor {
	return extendWithStrideImpl(x, shape, insertAt, stride)
}

func (x *Tensor) ExtendWithStride_(shape Shape, insertAt Axes, stride Stride) {
	*x = extendWithStrideImpl(*x, shape, insertAt, stride)
}

// concat

func concatImpl(x Tensor, y Tensor, axis uint) Tensor {
	node, err := wrapper.Concat(x.node, y.node, axis)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, y.err, err),
	}
}

func Concat(x Tensor, y Tensor, axis uint) Tensor {
	return concatImpl(x, y, axis)
}

func (x Tensor) Concat(y Tensor, axis uint) Tensor {
	return concatImpl(x, y, axis)
}

func (x *Tensor) Concat_(y Tensor, axis uint) {
	*x = concatImpl(*x, y, axis)
}

// expand

func expandImpl(x Tensor, axis uint, size uint) Tensor {
	node, err := wrapper.Expand(x.node, axis, size)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Expand(x Tensor, axis uint, size uint) Tensor {
	return expandImpl(x, axis, size)
}

func (x Tensor) Expand(axis uint, size uint) Tensor {
	return expandImpl(x, axis, size)
}

func (x *Tensor) Expand_(axis uint, size uint) {
	*x = expandImpl(*x, axis, size)
}

// repeat

func repeatImpl(x Tensor, repetitions Axes) Tensor {
	node, err := wrapper.Repeat(x.node, repetitions)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Repeat(x Tensor, repetitions Axes) Tensor {
	return repeatImpl(x, repetitions)
}

func (x Tensor) Repeat(repetitions Axes) Tensor {
	return repeatImpl(x, repetitions)
}

func (x *Tensor) Repeat_(repetitions Axes) {
	*x = repeatImpl(*x, repetitions)
}

// transpose

func transposeImpl(x Tensor, axes Axes) Tensor {
	node, err := wrapper.Transpose(x.node, axes)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Transpose(x Tensor, axes Axes) Tensor {
	return transposeImpl(x, axes)
}

func (x Tensor) Transpose(axes Axes) Tensor {
	return transposeImpl(x, axes)
}

func (x *Tensor) Transpose_(axes Axes) {
	*x = transposeImpl(*x, axes)
}

// TransposeFull acts like a traditional transpose where the order of all axis are reversed
func TransposeFull(x Tensor) Tensor {
	n := len(x.node.GetShape())
	ax := make(Axes, n)
	for i := 0; i < n; i++ {
		ax[i] = int(uint(n - i - 1))
	}
	return Transpose(x, ax)
}
