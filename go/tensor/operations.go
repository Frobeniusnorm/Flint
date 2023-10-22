package tensor

func (x Tensor) Add(y Tensor) Tensor {
	return x
}

func (x Tensor) Sub(y Tensor) Tensor {
	return x
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

// FIXME: or should this happen implicitly?
func (x Tensor) Convert() Tensor {
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

func (x Tensor) Transpose() Tensor {
	return x
}

func (x Tensor) Convolve() Tensor {
	return x
}

func (x Tensor) Slide() Tensor {
	return x
}

func (x Tensor) Index() Tensor {
	return x
}

func (x Tensor) IndexFromTensor(y Tensor) Tensor {
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

func (x Tensor) OneHot() Tensor {
	return x
}
