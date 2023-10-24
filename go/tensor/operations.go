package tensor

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type operation[T Numeric] interface {
	execute(x ...any) Tensor[T]
}

type lightOperation[T Numeric] func(x ...any) (T, error)

func (f lightOperation[T]) execute(x ...any) Tensor[T] {
	res, err := f(x...)
	// result composition and error handling
	var out Tensor[T]
	if err != nil {
		out.err = err
	} else {
		out.data = &res
	}
	out.init()
	return out
}

type nodeOperation[T Numeric] func(x ...any) (flint.GraphNode, error)

func (f nodeOperation[T]) execute(x ...any) Tensor[T] {
	flintNode, err := f(x...)
	// result composition and error handling
	var out Tensor[T]
	if err != nil {
		out.err = err
	} else {
		out.node = &flintNode
	}
	out.init()
	return out
}

func (x Tensor[T]) Add(y Tensor[T]) Tensor[T] {
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
	//var out Tensor[T]
	//if err != nil {
	//	out.err = err
	//} else {
	//	out.node = &res
	//}
	//out.init()
	//return out
}

func (x Tensor[T]) Sub(y Tensor[T]) Tensor[T] {
	if x.light() && y.light() {
		f := lightOperation[T](func(p ...any) (T, error) {
			x := p[0].(T)
			y := p[1].(T)
			return x - y, nil
		})
		return f.execute(x, y)
	} else {
		f := nodeOperation[T](func(p ...any) (flint.GraphNode, error) {
			x, y := p[0].(Tensor[T]), p[1].(Tensor[T]) // FIXME: this caused a bug - maybe variadic functions aren't that nice ...
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

func (x Tensor[T]) Mul(y Tensor[T]) Tensor[T] {
	return x
}

func (x Tensor[T]) Div(y Tensor[T]) Tensor[T] {
	return x
}

func (x Tensor[T]) Pow(y Tensor[T]) Tensor[T] {
	return x
}

func (x Tensor[T]) Log() Tensor[T] {
	return x
}

func (x Tensor[T]) Log2() Tensor[T] {
	return x
}

func (x Tensor[T]) Log10() Tensor[T] {
	return x
}

func (x Tensor[T]) Sqrt() Tensor[T] {
	return x
}

func (x Tensor[T]) Exp() Tensor[T] {
	return x
}

func (x Tensor[T]) Sin() Tensor[T] {
	return x
}

func (x Tensor[T]) Cos() Tensor[T] {
	return x
}

func (x Tensor[T]) Tan() Tensor[T] {
	return x
}

func (x Tensor[T]) Asin() Tensor[T] {
	return x
}

func (x Tensor[T]) Acos() Tensor[T] {
	return x
}

func (x Tensor[T]) Atan() Tensor[T] {
	return x
}

func (x Tensor[T]) Neg() Tensor[T] {
	return x
}

func (x Tensor[T]) Sign() Tensor[T] {
	return x
}

func (x Tensor[T]) Even() Tensor[T] {
	return x
}

func (x Tensor[T]) Equal(y Tensor[T]) Tensor[T] {
	return x
}

func (x Tensor[T]) Greater(y Tensor[T]) Tensor[T] {
	return x
}

func (x Tensor[T]) Less(y Tensor[T]) Tensor[T] {
	return x
}

func (x Tensor[T]) Matmul(y Tensor[T]) Tensor[T] {
	return x
}

func (x Tensor[T]) Flatten() Tensor[T] {
	// TODO: also flatten dim?
	return x
}

func (x Tensor[T]) Reshape() Tensor[T] {
	return x
}

func (x Tensor[T]) Minimum(y Tensor[T]) Tensor[T] {
	return x
}

func (x Tensor[T]) Maximum(y Tensor[T]) Tensor[T] {
	return x
}

func (x Tensor[T]) ReduceSum() Tensor[T] {
	return x
}

func (x Tensor[T]) ReduceMul() Tensor[T] {
	return x
}

func (x Tensor[T]) ReduceMin() Tensor[T] {
	return x
}

func (x Tensor[T]) ReduceMax() Tensor[T] {
	return x
}

func (x Tensor[T]) Slice() Tensor[T] {
	// TODO: what the good syntax here?
	return x
}

func (x Tensor[T]) SliceWithStride() Tensor[T] {
	return x
}

func (x Tensor[T]) Extend() Tensor[T] {
	return x
}

func (x Tensor[T]) ExtendWithStride() Tensor[T] {
	return x
}

func (x Tensor[T]) Concat(y Tensor[T]) Tensor[T] {
	return x
}

func (x Tensor[T]) Expand() Tensor[T] {
	return x
}

func (x Tensor[T]) Abs() Tensor[T] {
	return x
}

func (x Tensor[T]) Repeat() Tensor[T] {
	return x
}

func (x Tensor[T]) Transpose(axes Axes) Tensor[T] {
	return x
}

func (x Tensor[T]) Convolve() Tensor[T] {
	return x
}

func (x Tensor[T]) Slide() Tensor[T] {
	return x
}

func (x Tensor[T]) Index(indices Tensor[T]) Tensor[T] {
	x = x.readyNode()
	indices = indices.readyNode()

	flintNode, err := flint.Index(*x.node, *indices.node)
	var out Tensor[T]
	if err != nil {
		out.err = err
	} else {
		out.node = &flintNode
	}
	out.init()
	return out
}

func (x Tensor[T]) IndexFromTensor(y Tensor[T], indices Tensor[T]) Tensor[T] {
	return x
}

func (x Tensor[T]) SlidingWindow() Tensor[T] {
	return x
}

func (x Tensor[T]) Permute() Tensor[T] {
	return x
}

func (x Tensor[T]) Max() Tensor[T] {
	return x
}

func (x Tensor[T]) Min() Tensor[T] {
	return x
}

func (x Tensor[T]) Sum() Tensor[T] {
	return x
}

func (x Tensor[T]) OneHot(numClasses uint) Tensor[T] {
	return x
}
