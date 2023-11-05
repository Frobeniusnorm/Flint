package tensor

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type lightOperationUnary[T1 Numeric] func(x T1) Tensor

type lightOperationBinary[T1 Numeric, T2 Numeric] func(x T1, y T2) Tensor

func (f lightOperationBinary[T1, T2]) executeInner(x Tensor, y Tensor) Tensor {
	tt := largerType(x, y)
	switch tt {
	case flint.F_INT32:
		return f(lightVal[T1](x), lightVal[T2](y))
	case flint.F_INT64:
		return f(lightVal[int64](x), lightVal[int64](y))
	case flint.F_FLOAT32:
		return f(lightVal[float32](x), lightVal[float32](y))
	case flint.F_FLOAT64:
		return f(lightVal[float64](x), lightVal[float64](y))
	default:
		panic("invalid type")
	}
}

type lightOperationTernary[T1 Numeric, T2 Numeric, T3 Numeric] func(x T1, y T2, z T3) Tensor

//func (f lightOperation[T]) execute(x ...any) Tensor {
//	res, err := f(x...)
//	// result composition and error handling
//	var out Tensor
//	if err != nil {
//		out.err = err
//	} else {
//		out.data = &res
//	}
//	out.init()
//	return out
//}

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

func largerType(x Tensor, y Tensor) DataType {
	if x.dataType > y.dataType {
		return x.dataType
	} else {
		return y.dataType
	}
}

func lightLast(x Tensor, y Tensor) (a Tensor, b Tensor) {
	if !x.light() {
		return x, y
	} else if !y.light() {
		return y, x
	} else {
		return x.readyNode(), y
	}
}

func (x Tensor) Add(y Tensor) Tensor {
	if x.light() && y.light() {
		tt := largerType(x, y)
		switch tt {
		case flint.F_INT32:
			return Scalar(lightVal[int32](x) + lightVal[int32](y))
		case flint.F_INT64:
			return Scalar(lightVal[int64](x) + lightVal[int64](y))
		case flint.F_FLOAT32:
			return Scalar(lightVal[float32](x) + lightVal[float32](y))
		case flint.F_FLOAT64:
			return Scalar(lightVal[float64](x) + lightVal[float64](y))
		default:
			panic("invalid type")
		}
	} else {
		x, y = lightLast(x, y)
		if !y.light() {
			return FromNodeWithErr(flint.Add(*x.node, *y.node))
		}

		switch y.dataType {
		case flint.F_INT32:
			return FromNodeWithErr(flint.Add(*x.node, *y.dataInt32))
		case flint.F_INT64:
			return FromNodeWithErr(flint.Add(*x.node, *y.dataInt64))
		case flint.F_FLOAT32:
			return FromNodeWithErr(flint.Add(*x.node, *y.dataFloat32))
		case flint.F_FLOAT64:
			return FromNodeWithErr(flint.Add(*x.node, *y.dataFloat64))
		default:
			panic("invalid type")
		}
	}
}

func (x Tensor) Sub(y Tensor) Tensor {
	if x.light() && y.light() {
		tt := largerType(x, y)
		switch tt {
		case flint.F_INT32:
			return Scalar(lightVal[int32](x) - lightVal[int32](y))
		case flint.F_INT64:
			return Scalar(lightVal[int64](x) - lightVal[int64](y))
		case flint.F_FLOAT32:
			return Scalar(lightVal[float32](x) - lightVal[float32](y))
		case flint.F_FLOAT64:
			return Scalar(lightVal[float64](x) - lightVal[float64](y))
		default:
			panic("invalid type")
		}
	} else {
		if x.light() {
			switch y.dataType {
			case flint.F_INT32:
				return FromNodeWithErr(flint.Div(*x.node, *y.dataInt32))
			case flint.F_INT64:
				return FromNodeWithErr(flint.Div(*x.node, *y.dataInt64))
			case flint.F_FLOAT32:
				return FromNodeWithErr(flint.Div(*x.node, *y.dataFloat32))
			case flint.F_FLOAT64:
				return FromNodeWithErr(flint.Div(*x.node, *y.dataFloat64))
			default:
				panic("invalid type")
			}
		} else if y.light() {
			switch x.dataType {
			case flint.F_INT32:
				return FromNodeWithErr(flint.Div(*x.dataInt32, *y.node))
			case flint.F_INT64:
				return FromNodeWithErr(flint.Div(*x.dataInt64, *y.node))
			case flint.F_FLOAT32:
				return FromNodeWithErr(flint.Div(*x.dataFloat32, *y.node))
			case flint.F_FLOAT64:
				return FromNodeWithErr(flint.Div(*x.dataFloat64, *y.node))
			default:
				panic("invalid type")
			}
		} else {
			return FromNodeWithErr(flint.Div(*x.node, *y.node))
		}
	}
}

func (x Tensor) Mul(y Tensor) Tensor {
	if x.light() && y.light() {
		tt := largerType(x, y)
		switch tt {
		case flint.F_INT32:
			return Scalar(lightVal[int32](x) * lightVal[int32](y))
		case flint.F_INT64:
			return Scalar(lightVal[int64](x) * lightVal[int64](y))
		case flint.F_FLOAT32:
			return Scalar(lightVal[float32](x) * lightVal[float32](y))
		case flint.F_FLOAT64:
			return Scalar(lightVal[float64](x) * lightVal[float64](y))
		default:
			panic("invalid type")
		}
	} else {
		x, y = lightLast(x, y)
		if !y.light() {
			return FromNodeWithErr(flint.Mul(*x.node, *y.node))
		}

		switch y.dataType {
		case flint.F_INT32:
			return FromNodeWithErr(flint.Mul(*x.node, *y.dataInt32))
		case flint.F_INT64:
			return FromNodeWithErr(flint.Mul(*x.node, *y.dataInt64))
		case flint.F_FLOAT32:
			return FromNodeWithErr(flint.Mul(*x.node, *y.dataFloat32))
		case flint.F_FLOAT64:
			return FromNodeWithErr(flint.Mul(*x.node, *y.dataFloat64))
		default:
			panic("invalid type")
		}
	}
}

func (x Tensor) Div(y Tensor) Tensor {
	if x.light() && y.light() {
		tt := largerType(x, y)
		switch tt {
		case flint.F_INT32:
			return Scalar(lightVal[int32](x) / lightVal[int32](y))
		case flint.F_INT64:
			return Scalar(lightVal[int64](x) / lightVal[int64](y))
		case flint.F_FLOAT32:
			return Scalar(lightVal[float32](x) / lightVal[float32](y))
		case flint.F_FLOAT64:
			return Scalar(lightVal[float64](x) / lightVal[float64](y))
		default:
			panic("invalid type")
		}
	} else {
		if x.light() {
			switch y.dataType {
			case flint.F_INT32:
				return FromNodeWithErr(flint.Sub(*x.node, *y.dataInt32))
			case flint.F_INT64:
				return FromNodeWithErr(flint.Sub(*x.node, *y.dataInt64))
			case flint.F_FLOAT32:
				return FromNodeWithErr(flint.Sub(*x.node, *y.dataFloat32))
			case flint.F_FLOAT64:
				return FromNodeWithErr(flint.Sub(*x.node, *y.dataFloat64))
			default:
				panic("invalid type")
			}
		} else if y.light() {
			switch x.dataType {
			case flint.F_INT32:
				return FromNodeWithErr(flint.Sub(*x.dataInt32, *y.node))
			case flint.F_INT64:
				return FromNodeWithErr(flint.Sub(*x.dataInt64, *y.node))
			case flint.F_FLOAT32:
				return FromNodeWithErr(flint.Sub(*x.dataFloat32, *y.node))
			case flint.F_FLOAT64:
				return FromNodeWithErr(flint.Sub(*x.dataFloat64, *y.node))
			default:
				panic("invalid type")
			}
		} else {
			return FromNodeWithErr(flint.Sub(*x.node, *y.node))
		}
	}
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
