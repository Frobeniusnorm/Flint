package flint

import (
	"errors"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
)

// convolve

func convolveImpl(x Tensor, kernel Tensor, stride Stride) Tensor {
	node, err := wrapper.Convolve(x.node, kernel.node, stride)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, kernel.err, err),
	}
}

func Convolve(x Tensor, kernel Tensor, stride Stride) Tensor {
	return convolveImpl(x, kernel, stride)
}

func (x Tensor) Convolve(kernel Tensor, stride Stride) Tensor {
	return convolveImpl(x, kernel, stride)
}

func (x *Tensor) Convolve_(kernel Tensor, stride Stride) {
	*x = convolveImpl(*x, kernel, stride)
}

// index

func indexImpl(x Tensor, indices Tensor) Tensor {
	node, err := wrapper.Index(x.node, indices.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Index(x Tensor, indices Tensor) Tensor {
	return indexImpl(x, indices)
}

func (x Tensor) Index(indices Tensor) Tensor {
	return indexImpl(x, indices)
}

func (x *Tensor) Index_(indices Tensor) {
	*x = indexImpl(*x, indices)
}

// index set

func indexSetImpl(x Tensor, indices Tensor, y Tensor) Tensor {
	node, err := wrapper.IndexSet(x.node, indices.node, y.node)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, indices.err, y.err, err),
	}
}

func IndexSet(x Tensor, indices Tensor, y Tensor) Tensor {
	return indexSetImpl(x, indices, y)
}

func (x Tensor) IndexSet(indices Tensor, y Tensor) Tensor {
	return indexSetImpl(x, indices, y)
}

func (x *Tensor) IndexSet_(indices Tensor, y Tensor) {
	*x = indexSetImpl(*x, indices, y)
}

// sliding window

func slidingWindowImpl(x Tensor, window Shape, stride Stride) Tensor {
	node, err := wrapper.SlidingWindow(x.node, window, stride)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func SlidingWindow(x Tensor, window Shape, stride Stride) Tensor {
	return slidingWindowImpl(x, window, stride)
}

func (x Tensor) SlidingWindow(window Shape, stride Stride) Tensor {
	return slidingWindowImpl(x, window, stride)
}

func (x *Tensor) SlidingWindow_(window Shape, stride Stride) {
	*x = slidingWindowImpl(*x, window, stride)
}

// unslide window

func unslideWindowImpl(x Tensor, window Shape, stride Stride) Tensor {
	node, err := wrapper.UnslideWindow(x.node, window, stride)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func UnslideWindow(x Tensor, window Shape, stride Stride) Tensor {
	return unslideWindowImpl(x, window, stride)
}

func (x Tensor) UnslideWindow(window Shape, stride Stride) Tensor {
	return unslideWindowImpl(x, window, stride)
}

func (x *Tensor) UnslideWindow_(window Shape, stride Stride) {
	*x = unslideWindowImpl(*x, window, stride)
}

// permute

func permuteImpl(x Tensor, axes uint) Tensor {
	node, err := wrapper.Permute(x.node, axes)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Permute(x Tensor, axes uint) Tensor {
	return permuteImpl(x, axes)
}

func (x Tensor) Permute(axes uint) Tensor {
	return permuteImpl(x, axes)
}

func (x *Tensor) Permute_(axes uint) {
	*x = permuteImpl(*x, axes)
}

// dropout

func dropoutImpl(x Tensor, rate wrapper.Double) Tensor {
	node, err := wrapper.Dropout(x.node, rate)
	return Tensor{
		node: node,
		err:  errors.Join(x.err, err),
	}
}

func Dropout(x Tensor, rate wrapper.Double) Tensor {
	return dropoutImpl(x, rate)
}

func (x Tensor) Dropout(rate wrapper.Double) Tensor {
	return dropoutImpl(x, rate)
}

func (x *Tensor) Dropout_(rate wrapper.Double) {
	*x = dropoutImpl(*x, rate)
}
