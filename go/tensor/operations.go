package tensor

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
	"math"
)

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
	} else if y.light() {
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
	} else if x.light() {
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
	} else if y.light() {
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
	} else if x.light() {
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

func (x Tensor) Pow(y Tensor) Tensor {
	if x.light() && y.light() {
		tt := largerType(x, y)
		switch tt {
		case flint.F_INT32:
			return Scalar(int32(math.Pow(lightVal[float64](x), lightVal[float64](y))))
		case flint.F_INT64:
			return Scalar(int64(math.Pow(lightVal[float64](x), lightVal[float64](y))))
		case flint.F_FLOAT32:
			return Scalar(float32(math.Pow(lightVal[float64](x), lightVal[float64](y))))
		case flint.F_FLOAT64:
			return Scalar(float64(math.Pow(lightVal[float64](x), lightVal[float64](y))))
		default:
			panic("invalid type")
		}
	} else if y.light() {
		switch y.dataType {
		case flint.F_INT32:
			return FromNodeWithErr(flint.Pow(*x.node, *y.dataInt32))
		case flint.F_INT64:
			return FromNodeWithErr(flint.Pow(*x.node, *y.dataInt64))
		case flint.F_FLOAT32:
			return FromNodeWithErr(flint.Pow(*x.node, *y.dataFloat32))
		case flint.F_FLOAT64:
			return FromNodeWithErr(flint.Pow(*x.node, *y.dataFloat64))
		default:
			panic("invalid type")
		}
	} else if x.light() {
		x = x.readyNode()
		return FromNodeWithErr(flint.Pow(*x.node, *y.node))
	} else {
		return FromNodeWithErr(flint.Pow(*x.node, *y.node))
	}
}

func (x Tensor) Log() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Log(*x.node))
}

func (x Tensor) Log2() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Log2(*x.node))
}

func (x Tensor) Log10() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Log10(*x.node))
}

func (x Tensor) Sqrt() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Sqrt(*x.node))
}

func (x Tensor) Exp() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Exp(*x.node))
}

func (x Tensor) Sin() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Sin(*x.node))
}

func (x Tensor) Cos() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Cos(*x.node))
}

func (x Tensor) Tan() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Tan(*x.node))
}

func (x Tensor) Asin() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Asin(*x.node))
}

func (x Tensor) Acos() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Acos(*x.node))
}

func (x Tensor) Atan() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Atan(*x.node))
}

func (x Tensor) Neg() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Neg(*x.node))
}

func (x Tensor) Sign() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Sign(*x.node))
}

func (x Tensor) Even() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Even(*x.node))
}

func (x Tensor) Equal(y Tensor) Tensor {
	x = x.readyNode()
	y = y.readyNode()
	return FromNodeWithErr(flint.Equal(*x.node, *y.node))
}

func (x Tensor) Greater(y Tensor) Tensor {
	x = x.readyNode()
	y = y.readyNode()
	return FromNodeWithErr(flint.Greater(*x.node, *y.node))
}

func (x Tensor) Less(y Tensor) Tensor {
	x = x.readyNode()
	y = y.readyNode()
	return FromNodeWithErr(flint.Less(*x.node, *y.node))
}

func (x Tensor) Matmul(y Tensor) Tensor {
	x = x.readyNode()
	y = y.readyNode()
	return FromNodeWithErr(flint.Matmul(*x.node, *y.node))
}

func (x Tensor) Flatten() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Flatten(*x.node))
}

func (x Tensor) Reshape(shape Shape) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Reshape(*x.node, shape))
}

func (x Tensor) Minimum(y Tensor) Tensor {
	x = x.readyNode()
	y = y.readyNode()
	return FromNodeWithErr(flint.Minimum(*x.node, *y.node))
}

func (x Tensor) Maximum(y Tensor) Tensor {
	x = x.readyNode()
	y = y.readyNode()
	return FromNodeWithErr(flint.Maximum(*x.node, *y.node))
}

func (x Tensor) ReduceSum(dim int) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.ReduceSum(*x.node, dim))
}

func (x Tensor) ReduceMul(dim int) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.ReduceMul(*x.node, dim))
}

func (x Tensor) ReduceMin(dim int) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.ReduceMin(*x.node, dim))
}

func (x Tensor) ReduceMax(dim int) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.ReduceMax(*x.node, dim))
}

func (x Tensor) Slice(start Axes, end Axes) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Slice(*x.node, start, end))
}

func (x Tensor) SliceWithStride(start Axes, end Axes, stride Stride) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.SliceWithStride(*x.node, start, end, stride))
}

func (x Tensor) Extend(shape Shape, insertAt Axes) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Extend(*x.node, shape, insertAt))
}

func (x Tensor) ExtendWithStride(shape Shape, insertAt Axes, stride Stride) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.ExtendWithStride(*x.node, shape, insertAt, stride))
}

func (x Tensor) Concat(y Tensor, axis uint) Tensor {
	x = x.readyNode()
	y = y.readyNode()
	return FromNodeWithErr(flint.Concat(*x.node, *y.node, axis))
}

func (x Tensor) Expand(axis uint, size uint) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Expand(*x.node, axis, size))
}

func (x Tensor) Abs() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Abs(*x.node))
}

func (x Tensor) Repeat(repetitions Axes) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Repeat(*x.node, repetitions))
}

func (x Tensor) Transpose(axes Axes) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Transpose(*x.node, axes))
}

func (x Tensor) Convolve(kernel Tensor, stride Stride) Tensor {
	x = x.readyNode()
	kernel = kernel.readyNode()
	return FromNodeWithErr(flint.Convolve(*x.node, *kernel.node, stride))
}

func (x Tensor) Slide(kernel Tensor, stride Stride) Tensor {
	x = x.readyNode()
	kernel = kernel.readyNode()
	return FromNodeWithErr(flint.Slide(*x.node, *kernel.node, stride))
}

func (x Tensor) Index(indices Tensor) Tensor {
	x = x.readyNode()
	indices = indices.readyNode()
	return FromNodeWithErr(flint.Index(*x.node, *indices.node))
}

func (x Tensor) IndexFromTensor(y Tensor, indices Tensor) Tensor {
	x = x.readyNode()
	y = y.readyNode()
	indices = indices.readyNode()
	return FromNodeWithErr(flint.IndexSet(*x.node, *y.node, *indices.node))
}

func (x Tensor) SlidingWindow(size Shape, stride Stride) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.SlidingWindow(*x.node, size, stride))
}

func (x Tensor) Permute(axis uint) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Permute(*x.node, axis))
}

func (x Tensor) Max() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Max(*x.node))
}

func (x Tensor) Min() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Min(*x.node))
}

func (x Tensor) Sum() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.Sum(*x.node))
}

func (x Tensor) OneHot(numClasses uint) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(flint.OneHot(*x.node, numClasses))
}
