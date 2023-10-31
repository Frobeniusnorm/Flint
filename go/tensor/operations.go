package tensor

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type operation[T Numeric] interface {
	execute(x ...any) Tensor
}

type lightOperation[T Numeric] func(x ...any) (T, error)

func (f lightOperation[T]) execute(x ...any) Tensor {
	res, err := f(x...)
	// result composition and error handling
	var out Tensor
	if err != nil {
		out.err = err
	} else {
		out.data = &res
	}
	out.init()
	return out
}

type nodeOperation[T Numeric] func(x ...any) (flint.GraphNode, error)

func (f nodeOperation[T]) execute(x ...any) Tensor {
	flintNode, err := f(x...)
	// result composition and error handling
	var out Tensor
	if err != nil {
		out.err = err
	} else {
		out.node = &flintNode
	}
	out.init()
	return out
}

func (x Tensor) Add(y Tensor) Tensor {
	return x

	//if x.light() && y.light() {
	//	// FIXME!!!!
	//	op1 := (*x.data).(int32)
	//	op2 := (*y.data).(int64)
	//	return Scalar(op1 + op2)
	//}
	//x, y = lightLast(x, y)
	//
	//res, err := flint.Add(x.val().(flint.GraphNode), y.val().(int32)) // FIXME!!!
	//
	//var out Tensor
	//if err != nil {
	//	out.err = err
	//} else {
	//	out.node = &res
	//}
	//out.init()
	//return out
}

func (x Tensor) Sub(y Tensor) Tensor {
	if x.light() && y.light() {
		f := lightOperation[T](func(p ...any) (T, error) {
			x := p[0].(T)
			y := p[1].(T)
			return x - y, nil
		})
		return f.execute(x, y)
	} else {
		f := nodeOperation[T](func(p ...any) (flint.GraphNode, error) {
			x, y := p[0].(Tensor), p[1].(Tensor) // FIXME: this caused a bug - maybe variadic functions aren't that nice ...
			if x.light() {
				return flint.Sub(*x.data, *y.node)
			} else if y.light() {
				return flint.Sub(*x.node, *y.data)
			} else {
				return flint.Sub(*x.node, *y.node)
			}
		})
		return f.execute(x, y)
	}
}

func (x Tensor) Mul(y Tensor) Tensor {
	return x
}

func (x Tensor) Div(y Tensor) Tensor {
	return x
}

func (x Tensor) Pow(y Tensor) Tensor {
	return x
}

func (x Tensor) Log() Tensor {
	return x
}

func (x Tensor) Log2() Tensor {
	return x
}

func (x Tensor) Log10() Tensor {
	return x
}

func (x Tensor) Sqrt() Tensor {
	return x
}

func (x Tensor) Exp() Tensor {
	return x
}

func (x Tensor) Sin() Tensor {
	return x
}

func (x Tensor) Cos() Tensor {
	return x
}

func (x Tensor) Tan() Tensor {
	return x
}

func (x Tensor) Asin() Tensor {
	return x
}

func (x Tensor) Acos() Tensor {
	return x
}

func (x Tensor) Atan() Tensor {
	return x
}

func (x Tensor) Neg() Tensor {
	return x
}

func (x Tensor) Sign() Tensor {
	return x
}

func (x Tensor) Even() Tensor {
	return x
}

func (x Tensor) Equal(y Tensor) Tensor {
	return x
}

func (x Tensor) Greater(y Tensor) Tensor {
	return x
}

func (x Tensor) Less(y Tensor) Tensor {
	return x
}

func (x Tensor) Matmul(y Tensor) Tensor {
	return x
}

func (x Tensor) Flatten() Tensor {
	// TODO: also flatten dim?
	return x
}

func (x Tensor) Reshape() Tensor {
	return x
}

func (x Tensor) Minimum(y Tensor) Tensor {
	return x
}

func (x Tensor) Maximum(y Tensor) Tensor {
	return x
}

func (x Tensor) ReduceSum() Tensor {
	return x
}

func (x Tensor) ReduceMul() Tensor {
	return x
}

func (x Tensor) ReduceMin() Tensor {
	return x
}

func (x Tensor) ReduceMax() Tensor {
	return x
}

func (x Tensor) Slice() Tensor {
	// TODO: what the good syntax here?
	return x
}

func (x Tensor) SliceWithStride() Tensor {
	return x
}

func (x Tensor) Extend() Tensor {
	return x
}

func (x Tensor) ExtendWithStride() Tensor {
	return x
}

func (x Tensor) Concat(y Tensor) Tensor {
	return x
}

func (x Tensor) Expand() Tensor {
	return x
}

func (x Tensor) Abs() Tensor {
	return x
}

func (x Tensor) Repeat() Tensor {
	return x
}

func (x Tensor) Transpose(axes Axes) Tensor {
	return x
}

func (x Tensor) Convolve() Tensor {
	return x
}

func (x Tensor) Slide() Tensor {
	return x
}

func (x Tensor) Index(indices Tensor) Tensor {
	x = x.readyNode()
	indices = indices.readyNode()

	flintNode, err := flint.Index(*x.node, *indices.node)
	var out Tensor
	if err != nil {
		out.err = err
	} else {
		out.node = &flintNode
	}
	out.init()
	return out
}

func (x Tensor) IndexFromTensor(y Tensor, indices Tensor) Tensor {
	return x
}

func (x Tensor) SlidingWindow() Tensor {
	return x
}

func (x Tensor) Permute() Tensor {
	return x
}

func (x Tensor) Max() Tensor {
	return x
}

func (x Tensor) Min() Tensor {
	return x
}

func (x Tensor) Sum() Tensor {
	return x
}

func (x Tensor) OneHot(numClasses uint) Tensor {
	return x
}
