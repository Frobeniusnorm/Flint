package flint_old

import (
	"github.com/Frobeniusnorm/Flint/go/wrapper"
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
	if !x.isLight() {
		return x, y
	} else if !y.isLight() {
		return y, x
	} else {
		return x.readyNode(), y
	}
}

func (x Tensor) Add(y Tensor) Tensor {
	if x.isLight() && y.isLight() {
		tt := largerType(x, y)
		switch tt {
		case wrapper.F_INT32:
			return Scalar(lightVal[int32](x) + lightVal[int32](y))
		case wrapper.F_INT64:
			return Scalar(lightVal[int64](x) + lightVal[int64](y))
		case wrapper.F_FLOAT32:
			return Scalar(lightVal[float32](x) + lightVal[float32](y))
		case wrapper.F_FLOAT64:
			return Scalar(lightVal[float64](x) + lightVal[float64](y))
		default:
			panic("invalid type")
		}
	} else {
		x, y = lightLast(x, y)
		if !y.isLight() {
			return FromNodeWithErr(wrapper.Add(*x.node, *y.node))
		}

		switch y.dataType {
		case wrapper.F_INT32:
			return FromNodeWithErr(wrapper.Add(*x.node, *y.dataInt32))
		case wrapper.F_INT64:
			return FromNodeWithErr(wrapper.Add(*x.node, *y.dataInt64))
		case wrapper.F_FLOAT32:
			return FromNodeWithErr(wrapper.Add(*x.node, *y.dataFloat32))
		case wrapper.F_FLOAT64:
			return FromNodeWithErr(wrapper.Add(*x.node, *y.dataFloat64))
		default:
			panic("invalid type")
		}
	}
}

func (x Tensor) Sub(y Tensor) Tensor {
	if x.isLight() && y.isLight() {
		tt := largerType(x, y)
		switch tt {
		case wrapper.F_INT32:
			return Scalar(lightVal[int32](x) - lightVal[int32](y))
		case wrapper.F_INT64:
			return Scalar(lightVal[int64](x) - lightVal[int64](y))
		case wrapper.F_FLOAT32:
			return Scalar(lightVal[float32](x) - lightVal[float32](y))
		case wrapper.F_FLOAT64:
			return Scalar(lightVal[float64](x) - lightVal[float64](y))
		default:
			panic("invalid type")
		}
	} else if y.isLight() {
		switch y.dataType {
		case wrapper.F_INT32:
			return FromNodeWithErr(wrapper.Div(*x.node, *y.dataInt32))
		case wrapper.F_INT64:
			return FromNodeWithErr(wrapper.Div(*x.node, *y.dataInt64))
		case wrapper.F_FLOAT32:
			return FromNodeWithErr(wrapper.Div(*x.node, *y.dataFloat32))
		case wrapper.F_FLOAT64:
			return FromNodeWithErr(wrapper.Div(*x.node, *y.dataFloat64))
		default:
			panic("invalid type")
		}
	} else if x.isLight() {
		switch x.dataType {
		case wrapper.F_INT32:
			return FromNodeWithErr(wrapper.Div(*x.dataInt32, *y.node))
		case wrapper.F_INT64:
			return FromNodeWithErr(wrapper.Div(*x.dataInt64, *y.node))
		case wrapper.F_FLOAT32:
			return FromNodeWithErr(wrapper.Div(*x.dataFloat32, *y.node))
		case wrapper.F_FLOAT64:
			return FromNodeWithErr(wrapper.Div(*x.dataFloat64, *y.node))
		default:
			panic("invalid type")
		}
	} else {
		return FromNodeWithErr(wrapper.Div(*x.node, *y.node))
	}
}

func (x Tensor) Mul(y Tensor) Tensor {
	if x.isLight() && y.isLight() {
		tt := largerType(x, y)
		switch tt {
		case wrapper.F_INT32:
			return Scalar(lightVal[int32](x) * lightVal[int32](y))
		case wrapper.F_INT64:
			return Scalar(lightVal[int64](x) * lightVal[int64](y))
		case wrapper.F_FLOAT32:
			return Scalar(lightVal[float32](x) * lightVal[float32](y))
		case wrapper.F_FLOAT64:
			return Scalar(lightVal[float64](x) * lightVal[float64](y))
		default:
			panic("invalid type")
		}
	} else {
		x, y = lightLast(x, y)
		if !y.isLight() {
			return FromNodeWithErr(wrapper.Mul(*x.node, *y.node))
		}

		switch y.dataType {
		case wrapper.F_INT32:
			return FromNodeWithErr(wrapper.Mul(*x.node, *y.dataInt32))
		case wrapper.F_INT64:
			return FromNodeWithErr(wrapper.Mul(*x.node, *y.dataInt64))
		case wrapper.F_FLOAT32:
			return FromNodeWithErr(wrapper.Mul(*x.node, *y.dataFloat32))
		case wrapper.F_FLOAT64:
			return FromNodeWithErr(wrapper.Mul(*x.node, *y.dataFloat64))
		default:
			panic("invalid type")
		}
	}
}

func (x Tensor) Div(y Tensor) Tensor {
	if x.isLight() && y.isLight() {
		tt := largerType(x, y)
		switch tt {
		case wrapper.F_INT32:
			return Scalar(lightVal[int32](x) / lightVal[int32](y))
		case wrapper.F_INT64:
			return Scalar(lightVal[int64](x) / lightVal[int64](y))
		case wrapper.F_FLOAT32:
			return Scalar(lightVal[float32](x) / lightVal[float32](y))
		case wrapper.F_FLOAT64:
			return Scalar(lightVal[float64](x) / lightVal[float64](y))
		default:
			panic("invalid type")
		}
	} else if y.isLight() {
		switch y.dataType {
		case wrapper.F_INT32:
			return FromNodeWithErr(wrapper.Sub(*x.node, *y.dataInt32))
		case wrapper.F_INT64:
			return FromNodeWithErr(wrapper.Sub(*x.node, *y.dataInt64))
		case wrapper.F_FLOAT32:
			return FromNodeWithErr(wrapper.Sub(*x.node, *y.dataFloat32))
		case wrapper.F_FLOAT64:
			return FromNodeWithErr(wrapper.Sub(*x.node, *y.dataFloat64))
		default:
			panic("invalid type")
		}
	} else if x.isLight() {
		switch x.dataType {
		case wrapper.F_INT32:
			return FromNodeWithErr(wrapper.Sub(*x.dataInt32, *y.node))
		case wrapper.F_INT64:
			return FromNodeWithErr(wrapper.Sub(*x.dataInt64, *y.node))
		case wrapper.F_FLOAT32:
			return FromNodeWithErr(wrapper.Sub(*x.dataFloat32, *y.node))
		case wrapper.F_FLOAT64:
			return FromNodeWithErr(wrapper.Sub(*x.dataFloat64, *y.node))
		default:
			panic("invalid type")
		}
	} else {
		return FromNodeWithErr(wrapper.Sub(*x.node, *y.node))
	}
}

func (x Tensor) Pow(y Tensor) Tensor {
	if x.isLight() && y.isLight() {
		tt := largerType(x, y)
		switch tt {
		case wrapper.F_INT32:
			return Scalar(int32(math.Pow(lightVal[float64](x), lightVal[float64](y))))
		case wrapper.F_INT64:
			return Scalar(int64(math.Pow(lightVal[float64](x), lightVal[float64](y))))
		case wrapper.F_FLOAT32:
			return Scalar(float32(math.Pow(lightVal[float64](x), lightVal[float64](y))))
		case wrapper.F_FLOAT64:
			return Scalar(float64(math.Pow(lightVal[float64](x), lightVal[float64](y))))
		default:
			panic("invalid type")
		}
	} else if y.isLight() {
		switch y.dataType {
		case wrapper.F_INT32:
			return FromNodeWithErr(wrapper.Pow(*x.node, *y.dataInt32))
		case wrapper.F_INT64:
			return FromNodeWithErr(wrapper.Pow(*x.node, *y.dataInt64))
		case wrapper.F_FLOAT32:
			return FromNodeWithErr(wrapper.Pow(*x.node, *y.dataFloat32))
		case wrapper.F_FLOAT64:
			return FromNodeWithErr(wrapper.Pow(*x.node, *y.dataFloat64))
		default:
			panic("invalid type")
		}
	} else if x.isLight() {
		x = x.readyNode()
		return FromNodeWithErr(wrapper.Pow(*x.node, *y.node))
	} else {
		return FromNodeWithErr(wrapper.Pow(*x.node, *y.node))
	}
}

func (x Tensor) Log() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Log(*x.node))
}

func (x Tensor) Log2() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Log2(*x.node))
}

func (x Tensor) Log10() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Log10(*x.node))
}

func (x Tensor) Sqrt() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Sqrt(*x.node))
}

func (x Tensor) Exp() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Exp(*x.node))
}

func (x Tensor) Sin() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Sin(*x.node))
}

func (x Tensor) Cos() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Cos(*x.node))
}

func (x Tensor) Tan() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Tan(*x.node))
}

func (x Tensor) Asin() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Asin(*x.node))
}

func (x Tensor) Acos() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Acos(*x.node))
}

func (x Tensor) Atan() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Atan(*x.node))
}

func (x Tensor) Neg() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Neg(*x.node))
}

func (x Tensor) Sign() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Sign(*x.node))
}

func (x Tensor) Even() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Even(*x.node))
}

func (x Tensor) Equal(y Tensor) Tensor {
	x = x.readyNode()
	y = y.readyNode()
	return FromNodeWithErr(wrapper.Equal(*x.node, *y.node))
}

func (x Tensor) Greater(y Tensor) Tensor {
	x = x.readyNode()
	y = y.readyNode()
	return FromNodeWithErr(wrapper.Greater(*x.node, *y.node))
}

func (x Tensor) Less(y Tensor) Tensor {
	x = x.readyNode()
	y = y.readyNode()
	return FromNodeWithErr(wrapper.Less(*x.node, *y.node))
}

func (x Tensor) Matmul(y Tensor) Tensor {
	x = x.readyNode()
	y = y.readyNode()
	return FromNodeWithErr(wrapper.Matmul(*x.node, *y.node))
}

func (x Tensor) FlattenDim(axis int) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.FlattenDim(*x.node, axis))
}

func (x Tensor) Flatten() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Flatten(*x.node))
}

func (x Tensor) Reshape(shape Shape) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Reshape(*x.node, shape))
}

func (x Tensor) Minimum(y Tensor) Tensor {
	x = x.readyNode()
	y = y.readyNode()
	return FromNodeWithErr(wrapper.Minimum(*x.node, *y.node))
}

func (x Tensor) Maximum(y Tensor) Tensor {
	x = x.readyNode()
	y = y.readyNode()
	return FromNodeWithErr(wrapper.Maximum(*x.node, *y.node))
}

func (x Tensor) ReduceSum(dim int) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.ReduceSum(*x.node, dim))
}

func (x Tensor) ReduceMul(dim int) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.ReduceMul(*x.node, dim))
}

func (x Tensor) ReduceMin(dim int) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.ReduceMin(*x.node, dim))
}

func (x Tensor) ReduceMax(dim int) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.ReduceMax(*x.node, dim))
}

func (x Tensor) Slice(start Axes, end Axes) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Slice(*x.node, start, end))
}

func (x Tensor) SliceWithStride(start Axes, end Axes, stride Stride) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.SliceWithStride(*x.node, start, end, stride))
}

func (x Tensor) Extend(shape Shape, insertAt Axes) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Extend(*x.node, shape, insertAt))
}

func (x Tensor) ExtendWithStride(shape Shape, insertAt Axes, stride Stride) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.ExtendWithStride(*x.node, shape, insertAt, stride))
}

func (x Tensor) Concat(y Tensor, axis uint) Tensor {
	x = x.readyNode()
	y = y.readyNode()
	return FromNodeWithErr(wrapper.Concat(*x.node, *y.node, axis))
}

func (x Tensor) Expand(axis uint, size uint) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Expand(*x.node, axis, size))
}

func (x Tensor) Abs() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Abs(*x.node))
}

func (x Tensor) Repeat(repetitions Axes) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Repeat(*x.node, repetitions))
}

func (x Tensor) Transpose(axes Axes) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Transpose(*x.node, axes))
}

func (x Tensor) Convolve(kernel Tensor, stride Stride) Tensor {
	x = x.readyNode()
	kernel = kernel.readyNode()
	return FromNodeWithErr(wrapper.Convolve(*x.node, *kernel.node, stride))
}

func (x Tensor) Slide(kernel Tensor, stride Stride) Tensor {
	x = x.readyNode()
	kernel = kernel.readyNode()
	return FromNodeWithErr(wrapper.Slide(*x.node, *kernel.node, stride))
}

func (x Tensor) Index(indices Tensor) Tensor {
	x = x.readyNode()
	indices = indices.readyNode()
	return FromNodeWithErr(wrapper.Index(*x.node, *indices.node))
}

func (x Tensor) IndexFromTensor(y Tensor, indices Tensor) Tensor {
	x = x.readyNode()
	y = y.readyNode()
	indices = indices.readyNode()
	return FromNodeWithErr(wrapper.IndexSet(*x.node, *y.node, *indices.node))
}

func (x Tensor) SlidingWindow(size Shape, stride Stride) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.SlidingWindow(*x.node, size, stride))
}

func (x Tensor) Permute(axis uint) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Permute(*x.node, axis))
}

func (x Tensor) Max() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Max(*x.node))
}

func (x Tensor) Min() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Min(*x.node))
}

func (x Tensor) Sum() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Sum(*x.node))
}

func (x Tensor) OneHot(numClasses uint) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.OneHot(*x.node, numClasses))
}
