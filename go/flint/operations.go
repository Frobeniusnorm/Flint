package flint

// #include <flint/flint.h>
import "C"

/*
Add [operand1] and [operand2].
At least one of the parameters HAS to be a [GraphNode].
Addition is carried out element-wise.
*/
func Add[T baseNumeric | GraphNode](operand1 GraphNode, operand2 T) (GraphNode, error) {
	var flintNode *C.FGraphNode = nil
	var errno error

	switch c := any(operand2).(type) {
	case int32:
		flintNode, errno = C.fadd_ci(operand1.ref, C.int(c))
	case int64:
		flintNode, errno = C.fadd_cl(operand1.ref, C.long(c))
	case float32:
		flintNode, errno = C.fadd_cf(operand1.ref, C.float(c))
	case float64:
		flintNode, errno = C.fadd_cd(operand1.ref, C.double(c))
	case GraphNode:
		flintNode, errno = C.fadd_g(operand1.ref, c.ref)
	default:
		panic("invalid type")
	}

	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Pow takes [base] to the power of [exponent].
The [exponent] can be a constant number or another [GraphNode].
The operation is carried out element-wise.
*/
func Pow[T baseNumeric | GraphNode](base GraphNode, exponent T) (GraphNode, error) {
	var flintNode *C.FGraphNode = nil
	var errno error

	switch c := any(exponent).(type) {
	case GraphNode:
		flintNode, errno = C.fpow_g(base.ref, c.ref)
	case int32:
		flintNode, errno = C.fpow_ci(base.ref, C.int(c))
	case int64:
		flintNode, errno = C.fpow_cl(base.ref, C.long(c))
	case float32:
		flintNode, errno = C.fpow_cf(base.ref, C.float(c))
	case float64:
		flintNode, errno = C.fpow_cd(base.ref, C.double(c))
	default:
		panic("invalid type")
	}

	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Mul multiplies [operand1] by [operand2].
At least one of the parameters HAS to be a [GraphNode]
Multiplication is carried out element-wise.
*/
func Mul[T baseNumeric | GraphNode](operand1 GraphNode, operand2 T) (GraphNode, error) {
	var flintNode *C.FGraphNode = nil
	var errno error

	switch c := any(operand2).(type) {
	case GraphNode:
		flintNode, errno = C.fmul_g(operand1.ref, c.ref)
	case int32:
		flintNode, errno = C.fmul_ci(operand1.ref, C.int(c))
	case int64:
		flintNode, errno = C.fmul_cl(operand1.ref, C.long(c))
	case float32:
		flintNode, errno = C.fmul_cf(operand1.ref, C.float(c))
	case float64:
		flintNode, errno = C.fmul_cd(operand1.ref, C.double(c))
	default:
		panic("invalid type")
	}

	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Divide [numerator] by [denominator].
At least one of the parameters HAS to be a [GraphNode]
Division is carried out element-wise.
*/
func Divide[T1 baseNumeric | GraphNode, T2 baseNumeric | GraphNode](numerator T1, denominator T2) (GraphNode, error) {
	var flintNode *C.FGraphNode = nil
	var errno error

	switch x := any(numerator).(type) {
	case GraphNode:
		switch y := any(denominator).(type) {
		case GraphNode:
			flintNode, errno = C.fdiv_g(x.ref, y.ref)
		case int32:
			flintNode, errno = C.fdiv_ci(x.ref, C.int(y))
		case int64:
			flintNode, errno = C.fdiv_cl(x.ref, C.long(y))
		case float32:
			flintNode, errno = C.fdiv_cf(x.ref, C.float(y))
		case float64:
			flintNode, errno = C.fdiv_cd(x.ref, C.double(y))
		default:
			panic("invalid type")
		}
	case int32:
		if y, isNode := any(denominator).(GraphNode); isNode == true {
			flintNode, errno = C.fdiv_ici(C.int(x), y.ref)
		} else {
			panic("invalid type")
		}
	case int64:
		if y, isNode := any(denominator).(GraphNode); isNode == true {
			flintNode, errno = C.fdiv_icl(C.long(x), y.ref)
		} else {
			panic("invalid type")
		}
	case float32:
		if y, isNode := any(denominator).(GraphNode); isNode == true {
			flintNode, errno = C.fdiv_icf(C.float(x), y.ref)
		} else {
			panic("invalid type")
		}
	case float64:
		if y, isNode := any(denominator).(GraphNode); isNode == true {
			flintNode, errno = C.fdiv_icd(C.double(x), y.ref)
		} else {
			panic("invalid type")
		}
	default:
		panic("invalid type")
	}

	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Sub Subtracts [minuend] by [subtrahend].
At least one of the parameters HAS to be a [GraphNode].
Subtraction is carried out element-wise.
*/
func Sub[T1 baseNumeric | GraphNode, T2 baseNumeric | GraphNode](minuend T1, subtrahend T2) (GraphNode, error) {
	var flintNode *C.FGraphNode = nil
	var errno error

	switch x := any(minuend).(type) {
	case GraphNode:
		switch y := any(subtrahend).(type) {
		case GraphNode:
			flintNode, errno = C.fsub_g(x.ref, y.ref)
		case int32:
			flintNode, errno = C.fsub_ci(x.ref, C.int(y))
		case int64:
			flintNode, errno = C.fsub_cl(x.ref, C.long(y))
		case float32:
			flintNode, errno = C.fsub_cf(x.ref, C.float(y))
		case float64:
			flintNode, errno = C.fsub_cd(x.ref, C.double(y))
		default:
			panic("invalid type")
		}
	case int32:
		if y, isNode := any(subtrahend).(GraphNode); isNode == true {
			flintNode, errno = C.fsub_ici(C.int(x), y.ref)
		} else {
			panic("invalid type")
		}
	case int64:
		if y, isNode := any(subtrahend).(GraphNode); isNode == true {
			flintNode, errno = C.fsub_icl(C.long(x), y.ref)
		} else {
			panic("invalid type")
		}
	case float32:
		if y, isNode := any(subtrahend).(GraphNode); isNode == true {
			flintNode, errno = C.fsub_icf(C.float(x), y.ref)
		} else {
			panic("invalid type")
		}
	case float64:
		if y, isNode := any(subtrahend).(GraphNode); isNode == true {
			flintNode, errno = C.fsub_icd(C.double(x), y.ref)
		} else {
			panic("invalid type")
		}
	default:
		panic("invalid type")
	}

	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

// Log takes the element wise logarithm naturalis of x.
func Log(x GraphNode) (GraphNode, error) {
	flintNode, errno := C.flog(x.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

// Log2 takes the element wise base 10 logarithm of x.
func Log2(x GraphNode) (GraphNode, error) {
	flintNode, errno := C.flog2(x.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

// Log10 takes the element wise base 10 logarithm of x.
func Log10(x GraphNode) (GraphNode, error) {
	flintNode, errno := C.flog10(x.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

// Sin takes the element wise sinus of x.
func Sin(x GraphNode) (GraphNode, error) {
	flintNode, errno := C.fsin(x.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

// Sqrt takes the element wise square root of x.
func Sqrt(x GraphNode) (GraphNode, error) {
	flintNode, errno := C.fsqrt_g(x.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

// Exp takes each element as the exponent for power of base e.
func Exp(x GraphNode) (GraphNode, error) {
	flintNode, errno := C.fexp(x.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

// Cos takes the element wise cosine of x.
func Cos(x GraphNode) (GraphNode, error) {
	flintNode, errno := C.fcos(x.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

// Tan takes the element wise tangent of x.
func Tan(x GraphNode) (GraphNode, error) {
	flintNode, errno := C.ftan(x.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

// Asin takes the element wise inverse sinus of x.
func Asin(x GraphNode) (GraphNode, error) {
	flintNode, errno := C.fasin(x.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

// Acos takes the element wise inverse cosine of x.
func Acos(x GraphNode) (GraphNode, error) {
	flintNode, errno := C.facos(x.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

// Atan takes the element wise inverse tangent of x.
func Atan(x GraphNode) (GraphNode, error) {
	flintNode, errno := C.fatan(x.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

// Neg swaps the sign of each element.
func Neg(x GraphNode) (GraphNode, error) {
	flintNode, errno := C.fneg(x.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Sign applies the sign function to each element.
i.e. x[i] = 1 if x[i] >= 0 else x[i] = -1
The input tensor [x] must have an integer type.
This function returns a [F_INT32] [GraphNode].
*/
func Sign(x GraphNode) (GraphNode, error) {
	flintNode, errno := C.fsign(x.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Even gives the result of module 2 for each element.
i.e. x[i] = 1 if x[i] mod 2 == 0 else x[i] = 0
This function returns a [F_INT32] [GraphNode].
*/
func Even(x GraphNode) (GraphNode, error) {
	flintNode, errno := C.feven(x.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Equal compares a tensor and a constant elementwise by [a] = [b] and returns a 0,1 [F_INT32] [GraphNode].
*/
func Equal[T baseNumeric | GraphNode](a GraphNode, b T) (GraphNode, error) {
	var flintNode *C.FGraphNode = nil
	var errno error

	switch c := any(b).(type) {
	case GraphNode:
		flintNode, errno = C.fequal_g(a.ref, c.ref)
	case int32:
		flintNode, errno = C.fequal_ci(a.ref, C.int(c))
	case int64:
		flintNode, errno = C.fequal_cl(a.ref, C.long(c))
	case float32:
		flintNode, errno = C.fequal_cf(a.ref, C.float(c))
	case float64:
		flintNode, errno = C.fequal_cd(a.ref, C.double(c))
	default:
		panic("invalid type")
	}

	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Greater compares a tensor and a constant elementwise by [a] > [b] and returns a 0,1 [F_INT32] [GraphNode].
*/
func Greater[T baseNumeric | GraphNode](a GraphNode, b T) (GraphNode, error) {
	var flintNode *C.FGraphNode = nil
	var errno error

	switch c := any(b).(type) {
	case GraphNode:
		flintNode, errno = C.fgreater_g(a.ref, c.ref)
	case int32:
		flintNode, errno = C.fgreater_ci(a.ref, C.int(c))
	case int64:
		flintNode, errno = C.fgreater_cl(a.ref, C.long(c))
	case float32:
		flintNode, errno = C.fgreater_cf(a.ref, C.float(c))
	case float64:
		flintNode, errno = C.fgreater_cd(a.ref, C.double(c))
	default:
		panic("invalid type")
	}

	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Less compares a tensor and a constant elementwise by [a] < [b] and returns a 0,1 [F_INT32] [GraphNode].
*/
func Less[T baseNumeric | GraphNode](a GraphNode, b T) (GraphNode, error) {
	var flintNode *C.FGraphNode = nil
	var errno error

	switch c := any(b).(type) {
	case GraphNode:
		flintNode, errno = C.fless_g(a.ref, c.ref)
	case int32:
		flintNode, errno = C.fless_ci(a.ref, C.int(c))
	case int64:
		flintNode, errno = C.fless_cl(a.ref, C.long(c))
	case float32:
		flintNode, errno = C.fless_cf(a.ref, C.float(c))
	case float64:
		flintNode, errno = C.fless_cd(a.ref, C.double(c))
	default:
		panic("invalid type")
	}
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Matmul carries out matrix multiplication on the last two dimensions of the tensors.

E.g. a matrix multiplication of two tensors with shapes (64, 32, 16) and (16, 24) will yield a tensor with shape (64, 32, 24).
Since for one entry of the tensor multiple other previous entries are needed, the operand tensors need to be executed first.
Therefor the method will implicitly (or eagerly) execute the two parameter nodes [a] and [b] if their data is not already present.
*/
func Matmul(a GraphNode, b GraphNode) (GraphNode, error) {
	flintNode, errno := C.fmatmul(a.ref, b.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Flatten the complete tensor to a tensor with one dimension.
E.g:

	Flatten([[[3, 1, 4], [2, 1, 5]], [[0, 4, 2], [4, 7, 9]]]) =
		[3, 1, 4, 2, 1, 5, 0, 4, 2, 4, 7, 9].
*/
func Flatten(a GraphNode) (GraphNode, error) {
	flintNode, errno := C.fflatten(a.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
FlattenDim flattens a tensor [a] with n dimensions along dimension [dim], resulting in a tensor with n-1 dimensions.
Flattening a dimension will remove it from the shape of the tensor, therefor it's not possible to flatten the dimension 0.

E.g:

	FlattenDim([[[3, 1, 4], [2, 1, 5]], [[0, 4, 2], [4, 7, 9]]], 1) =
		[[3,1,4], [2,1,5], [0,4,2], [4,7,9]]
*/
func FlattenDim(a GraphNode, dim int) (GraphNode, error) {
	flintNode, errno := C.fflatten_dimension(a.ref, C.int(dim))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Convert the data of [a] to the type given by [newType].
*/
func Convert(a GraphNode, newType DataType) (GraphNode, error) {
	flintNode, errno := C.fconvert(a.ref, C.enum_FType(newType))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Reshape the underlying data of the tensor to the new shape.
The product of each dimension of the new shape must be the same as the product of the dimensions of the previous shape.
This means it must describe the same number of entries of the tensor.
*/
func Reshape(a GraphNode, shape Shape) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)

	flintNode, errno := C.freshape(a.ref, &(newShape[0]), C.int(len(shape)))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Minimum takes the minimum of two tensors (or a tensor and value) element wise along the last dimension of each.
*/
func Minimum[T baseNumeric | GraphNode](a GraphNode, b T) (GraphNode, error) {
	var flintNode *C.FGraphNode = nil
	var errno error

	switch c := any(b).(type) {
	case GraphNode:
		flintNode, errno = C.fmin_g(a.ref, c.ref)
	case int32:
		flintNode, errno = C.fmin_ci(a.ref, C.int(c))
	case int64:
		flintNode, errno = C.fmin_cl(a.ref, C.long(c))
	case float32:
		flintNode, errno = C.fmin_cf(a.ref, C.float(c))
	case float64:
		flintNode, errno = C.fmin_cd(a.ref, C.double(c))
	default:
		panic("invalid type")
	}
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Maximum takes the maximum of two tensors (or a tensor and value) element wise along the last dimension of each.
*/
func Maximum[T baseNumeric | GraphNode](a GraphNode, b T) (GraphNode, error) {
	var flintNode *C.FGraphNode = nil
	var errno error

	switch c := any(b).(type) {
	case GraphNode:
		flintNode, errno = C.fmax_g(a.ref, c.ref)
	case int32:
		flintNode, errno = C.fmax_ci(a.ref, C.int(c))
	case int64:
		flintNode, errno = C.fmax_cl(a.ref, C.long(c))
	case float32:
		flintNode, errno = C.fmax_cf(a.ref, C.float(c))
	case float64:
		flintNode, errno = C.fmax_cd(a.ref, C.double(c))
	default:
		panic("invalid type")
	}
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
ReduceSum reduces one dimension of the tensor by additive folding.

E.g:

	ReduceSum([[1,2,3], [4,5,6]], 0) = [5,7,9]
	ReduceSum([[1,2,3], [4,5,6]], 1) = [6,15]

The results of the predecessor node must be available, to
ensure that the method may execute the parameter node.
*/
func ReduceSum(a GraphNode, dim int) (GraphNode, error) {
	flintNode, errno := C.freduce_sum(a.ref, C.int(dim))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
ReduceMul reduces one dimension of the tensor by multiplicative folding.

E.g:

	ReduceMul([[1,2,3], [4,5,6]], 0) = [4,10,18]
	ReduceMul([[1,2,3], [4,5,6]], 1) = [6, 120]

The results of the predecessor node must be available; to ensure that the method may execute the parameter node.
*/
func ReduceMul(a GraphNode, dim int) (GraphNode, error) {
	flintNode, errno := C.freduce_mul(a.ref, C.int(dim))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
ReduceMin reduces one dimension of the tensor by keeping the minimum.

E.g:

	ReduceMin([[1,32,3], [4,5,3]], 0) = [1,5,3]
	ReduceMin([[9,2,3], [-1,5,6]], 1) = [2, -1]

The results of the predecessor node must be available; to ensure that the method may execute the parameter node.
*/
func ReduceMin(a GraphNode, dim int) (GraphNode, error) {
	flintNode, errno := C.freduce_min(a.ref, C.int(dim))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
ReduceMax reduces one dimension of the tensor by keeping the maximum.

E.g:

	ReduceMax([[1,32,3], [4,5,3]], 0) = [4,32,3]
	ReduceMax([[9,2,3], [-1,5,6]], 1) = [9, 6]

The results of the predecessor node must be available; to ensure that the method may execute the parameter node.
*/
func ReduceMax(a GraphNode, dim int) (GraphNode, error) {
	flintNode, errno := C.freduce_max(a.ref, C.int(dim))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Slice selects a slice of the tensor with a dimension wise start and end index.
[start] and [end] are arrays with as many entries as the tensor has dimensions.
They may contain negative values, which are then subtracted from the end of the tensor
(e.g. -1 means the element before the last element).
[start] is inclusive and describes the start index of the selection per dimension and [end] describes the end index per dimension and is exclusive.
*/
func Slice(a GraphNode, start Axes, end Axes) (GraphNode, error) {
	newStart := convertArray[uint, C.long](start)
	newEnd := convertArray[uint, C.long](end)

	flintNode, errno := C.fslice(a.ref, &(newStart[0]), &(newEnd[0]))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
SliceWithStride selects a slice of the tensor [node] with a dimension wise start index, end index and stride.
[start], [end] and [stride] are arrays with as many entries as the tensor has dimensions.
[start] and [end] may contain negative values, which are then subtracted from the end of the tensor
(e.g. -1 means the element before the last element).
[start] is inclusive and describes the start index of the selection per dimension and [end] describes the end index per dimension and is exclusive.
[stride] contains the per dimension step size (e.g. 2 meaning every second element will be selected etc.) and may be negative as well,
which reverses the traversal order (the first elements are selected as the last ones).
For a negative stride, [start] > [end] must hold (for a positive of course [end] > [start]) for each dimension.
*/
func SliceWithStride(node GraphNode, start Axes, end Axes, stride Stride) (GraphNode, error) {
	// TODO: check that axes has the right length compared to node! (or does the C function do this?)
	newStart := convertArray[uint, C.long](start)
	newEnd := convertArray[uint, C.long](end)
	newStride := convertArray[int, C.long](stride)

	flintNode, errno := C.fslice_step(node.ref, &(newStart[0]), &(newEnd[0]), &(newStride[0]))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Extend creates a new tensor of zeroes with the requested shape.
The original tensor is embedded at the given indices.
  - [node]: original tensor which shape is to be extended
  - [shape]: array of new sizes per dimension. Has the same number of entries and the shape of [node].
  - [insertAt]: array with indices per dimension of [node], denoting where [node] is to be placed in the resulting tensor
*/
func Extend(node GraphNode, shape Shape, insertAt Axes) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)
	newInsertAt := convertArray[uint, C.size_t](insertAt)

	flintNode, errno := C.fextend(node.ref, &(newShape[0]), &(newInsertAt[0]))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
ExtendWithStride creates a new tensor of zeroes with the requested shape.
The original tensor is embedded at the given indices.
  - [node]: original tensor which shape is to be extended,
  - [shape]: array of new sizes per dimension. Has the same number of entries as [node] has dimensions.
  - [insertAt]: array with indices per dimension denoting where [node] is to be placed in the resulting tensor. Has a value per dimension of [node].
  - [stride]: allows to pull apart [node], em-placing zeros between each value of [node]. Has a value per dimension.
*/
func ExtendWithStride(node GraphNode, shape Shape, insertAt Axes, stride Stride) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)
	newInsertAt := convertArray[uint, C.size_t](insertAt)
	newStride := convertArray[int, C.long](stride)

	flintNode, errno := C.fextend_step(node.ref, &(newShape[0]), &(newInsertAt[0]), &(newStride[0]))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Concat two nodes ([nodeA], [nodeB]) with each other along an [axis].
The nodes have to have the same type and dimensions.

E.g:

	Concat({[[0, 1], [2, 3]], [[4, 5], [6, 7]]}, 0) = [[0, 1], [2, 3], [4, 5], [6, 7]]

	Concat({[[0, 1], [2, 3]], [[4, 5], [6, 7]]}, 1) = [[0, 1, 4, 5], [2, 3, 6, 7]]
*/
func Concat(nodeA GraphNode, nodeB GraphNode, axis uint) (GraphNode, error) {
	flintNode, errno := C.fconcat(nodeA.ref, nodeB.ref, C.uint(axis))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Expand adds a new dimension at an arbitrary position to the tensor and repeats the following dimensions to match a given shape.
  - [axis]: the dimension prior to which the new dimension will be inserted (0 means a new dimension in the front, n means as a new last dimension).
  - [size]: the new size of that dimension (repeats the following dimensions ax_size - 1 times).
*/
func Expand(a GraphNode, axis uint, size uint) (GraphNode, error) {
	flintNode, errno := C.fexpand(a.ref, C.uint(axis), C.uint(size))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Abs takes the elementwise absolute value of [node], i.e. |a[i]|
*/
func Abs(node GraphNode) (GraphNode, error) {
	flintNode, errno := C.fabs_g(node.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Repeat dimensions of a tensor multiple times.
  - [node]: the node in which dimensions are to be repeated
  - [axes]: array with the same number of entries as the tensor has dimensions

E.g:

	Repeat([[0,1], [2,3]], [2, 3]) =
		[[0,1,0,1,0,1], [2,3,2,3,2,3], [0,1,0,1,0,1], [2,3,2,3,2,3]]
*/
func Repeat(node GraphNode, repetitions Axes) (GraphNode, error) {
	newRepetitions := convertArray[uint, C.int](repetitions)

	flintNode, errno := C.frepeat(node.ref, &(newRepetitions[0]))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Transpose tensor [node] along multiple dimensions.
The array [axes] has the same number of entries as [node] has dimensions, which gives the permutations of dimensions.

The tensor will have a resulting shape in which the size each dimension corresponds to the former size in dimension in [axes].
*/
func Transpose(node GraphNode, axes Axes) (GraphNode, error) {
	newAxes := convertArray[uint, C.int](axes)

	flintNode, errno := C.ftranspose(node.ref, &(newAxes[0]))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Convolve the n-dimensional input tensor [node] with an n-dimensional filter
[kernel] and a per dimensional [stride] with size of n-1.
It is expected that [node] and [kernel] have the same size in their last dimension (which will be completely reduced by the convolution).
In all other dimensions the size of [node] should be larger or equal to the size of [kernel].
The kernel will be 'slid' over [node] in each dimension, multiplying all
values of [kernel] with the corresponding ones in [node] and summing them up to
a single value and moving the kernel further by the value given in [stride] in that corresponding dimension.

The implementation does not include any padding, meaning only convolutions where the complete kernel still fits into the array will be executed (the shape will be calculated correspondingly).
If you want to modify this behaviour (i.e. include padding) you can use [Extend], [Slice], or similar.

The resulting [GraphNode] will therefore have a shape with dimensionality n - 1 and size of:

	(shape[i] - kernel.get_shape()[i] - 1) / stride[i]
	if (shape[i] - kernel.get_shape()[i] - 1) is dividable by stride[i]
	else (shape[i] - kernel.get_shape()[i] - 1) / stride[i] + 1
*/
func Convolve(node GraphNode, kernel GraphNode, stride Stride) (GraphNode, error) {
	newStride := convertArray[int, C.uint](stride)

	flintNode, errno := C.fconvolve(node.ref, kernel.ref, &(newStride[0]))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Slide moves the [kernel] along the node, multiplying it with the elements of the [node] it is slid over.
For each element all multiplied values are summed up, so that the result has the same shape as the [kernel].
Every element in the result is the accumulated sum of the product of that element with all elements it was slid over
The [kernel] is initially placed so that the first element of [node] and the first element of [kernel] overlap.
It is then moved with the [stride] value for each dimension except for the last, just like it would be by [Convolve].
[stride] should have 1 dimension less than [node] and [kernel].
With the difference, that everything is accumulated for the [kernel] instead of the original node.

The last dimension of [node] and [kernel] should be equal,
therefore it has no stride in that dimension since the complete kernel is multiplied in that dimension.
*/
func Slide(node GraphNode, kernel GraphNode, stride Stride) (GraphNode, error) {
	newStride := convertArray[int, C.uint](stride)

	flintNode, errno := C.fslide(node.ref, kernel.ref, &(newStride[0]))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Index selects single elements with an index-tensor (integer tensor containing indices for the selected dimension).
It indexes a dimension of the input tensor and the result has the shape of the input tensor except for the indexed dimension.
It is assumed that except for the last entry its shape is a prefix of the shape of the input tensor and the indexing will occur in the matched subsets.
The last dimension of indices is the one indexed in node.

Take the "subset" of the matrix where the first two slices are swapped:

	Index([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
			[1, 0]) =
		[[[4, 5], [6, 7]], [[0, 1], [2, 3]]]

TODO description:

	Index([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
			[0, 0, 1]) =
		[[[0, 1], [0, 1], [1, 2]], [[3, 4], [3, 4], [5, 6]], [[7, 8], [7, 8], [9, 10]]]

TODO description:

	Index([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
			[[0], [1], [0]]) =
		[[[0], [2]], [[5], [7]], [[8], [10]]]

TODO description:

	Index([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
			[[0, 0], [1, 0], [0, 1]]) =
		[[[0, 0], [2, 2]], [[5, 4], [7, 6]], [[8, 9], [10, 11]]]
*/
func Index(node GraphNode, indices GraphNode) (GraphNode, error) {
	flintNode, errno := C.findex(node.ref, indices.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
IndexSet Assigns to each element in nodeB one element in nodeA where that element will be
"send" to, i.e. for the place in nodeA, the index pointer will be set to the
corresponding element from nodeB. If multiple elements from nodeB are sent to the
same place in nodeA they will be summed up.
The shape of indices must be a prefix of the shape of nodeB,
meaning it can have as many dimensions as nodeB or less,
but the sizes of the dimensions must be the same as the first of the shape of nodeB.

Adds the first ([4,5]) and second ([6,7]) row of nodeB into nodeA, as specified by indices ([0,0,2])
the last row ([8,9]) instead replaces the previous data:

	IndexSet([[0, 1], [2, 3], [4, 5], [6, 7]],
				[[4, 5], [6, 7], [8, 9]],
				[0, 0, 2]) =
		[[10, 12], [2, 3], [8, 9], [6, 7]]

Instead of specifying indices per row, we can also define them per element.
In this case an index of -1 means to discard the element from nodeB and keep the entry of nodeA:

	IndexSet([[0, 1], [2, 3], [4, 5], [6, 7]],
	            [[4, 5], [6, 7], [8, 9], [10, 11]],
	            [[-1, 0], [1, 1], [1, 0], [1, -1]]) =
		[[5, 1], [2, 13], [9, 8], [6, 10]]
*/
func IndexSet(nodeA GraphNode, nodeB GraphNode, indices GraphNode) (GraphNode, error) {
	flintNode, errno := C.findex_set(nodeA.ref, nodeB.ref, indices.ref)
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
SlidingWindow moves a window view with size along the node by starting
with aligning the first element of the view with the first element of the node,
copying the elements of the view and moving the window by the stride
given for each dimension (the window is first moved in the innermost
dimension and after each is iterated moves it in the outer dimensions).
Each view becomes a new element in a new outer dimension.

This example moves a 3x2 rectangle across the node, each time taking one stride in each direction:

	SlidingWindow([[0, 1], [2, 3], [4, 5], [6, 7]], [3, 2], [1, 1]) =
		[[[0, 1], [2, 3], [4, 5]], [[2, 3], [4, 5], [6, 7]]]

This example moves a 2x2x2 cube across the node, this time moving 2 across the first and last axis for each stride:

	SlidingWindow([[[0,1,2],[1,2,3],[2,3,4]],
	                 [[1,2,3],[2,3,4],[3,4,5]],
	                 [[2,3,4],[3,4,5],[4,5,6]],
	                 [[3,4,5],[4,5,6],[5,6,7]]],
	                 [2, 2, 2], [2, 1, 2]) =
		[[[[0, 1], [1, 2]],
		  [[1, 2], [2, 3]]],
		 [[[1, 2], [2, 3]],
		  [[2, 3], [3, 4]]],
		 [[[2, 3], [3, 4]],
		  [[3, 4], [4, 5]]],
		 [[[3, 4], [4, 5]],
		  [[4, 5], [5, 6]]]]
*/
func SlidingWindow(node GraphNode, size Shape, stride Stride) (GraphNode, error) {
	newSize := convertArray[uint, C.size_t](size)
	newStride := convertArray[int, C.uint](stride)

	flintNode, errno := C.fsliding_window(node.ref, &(newSize[0]), &(newStride[0]))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
Permute randomly permutes (= swaps multiple elements with each other without creating, copying or deleting new ones) one axis of the input tensor.
*/
func Permute(a GraphNode, axis uint) (GraphNode, error) {
	flintNode, errno := C.fpermutate(a.ref, C.uint(axis))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

// Max returns the maximum value across all dimensions
// see: https://numpy.org/doc/stable/reference/generated/numpy.max.html
// FIXME: func Max(a GraphNode, axes Axis, scalar bool, keepDims bool) {
func Max(node GraphNode) (GraphNode, error) {
	//for !node.GetShape().Equal(Shape{1}) { }
	node, err := Flatten(node)
	if err != nil {
		return GraphNode{}, err
	}

	node, err = ReduceMax(node, 0)
	if err != nil {
		return GraphNode{}, err
	}

	return node, nil
}

// Min returns the minimum value across all dimensions
func Min(node GraphNode) (GraphNode, error) {
	node, err := Flatten(node)
	if err != nil {
		return GraphNode{}, err
	}

	node, err = ReduceMin(node, 0)
	if err != nil {
		return GraphNode{}, err
	}

	return node, nil
}

// Sum sums up all values in the flattened tensor
func Sum(node GraphNode) (GraphNode, error) {
	node, err := Flatten(node)
	if err != nil {
		return GraphNode{}, err
	}

	node, err = ReduceSum(node, 0)
	if err != nil {
		return GraphNode{}, err
	}

	return node, nil
}

// TransposeFull acts like a traditional transpose where the order of all axis are reversed
func TransposeFull(node GraphNode) (GraphNode, error) {
	n := len(node.GetShape())
	ax := make(Axes, n)
	for i := 0; i < n; i++ {
		ax[i] = uint(n - i - 1)
	}
	return Transpose(node, ax)
}

func OneHot(node GraphNode, numClasses uint) (GraphNode, error) {
	id, err := CreateGraphIdentity(numClasses)
	if err != nil {
		return GraphNode{}, err
	}

	id, err = Index(id, node)
	if err != nil {
		return GraphNode{}, err
	}

	return id, nil
}
