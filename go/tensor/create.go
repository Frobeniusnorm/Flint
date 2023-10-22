package tensor

func Create(data any) Tensor {
	// if scalar type just keep it, don't turn into tensor yet!
	// if nd array flatten it a generate
	// if 1d array generate normally
	// if f graph node, don;t just set reference counter
	return Tensor{}
}

func Scalar() Tensor {
	return Tensor{}
}

func Constant() Tensor {
	return Tensor{}
}

func Random() Tensor {
	return Tensor{}
}

func Arrange() Tensor {
	return Tensor{}
}

func Identity() Tensor {
	return Tensor{}

}
