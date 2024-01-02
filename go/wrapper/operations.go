package wrapper

// #include <flint/flint.h>
import "C"

func AddGraphGraph(a GraphNode, b GraphNode) (GraphNode, error) {
	return returnHelper(C.fadd_g(a.ref, b.ref))
}

func AddGraphInt(a GraphNode, b Int) (GraphNode, error) {
	return returnHelper(C.fadd_ci(a.ref, C.int(b)))
}

func AddGraphLong(a GraphNode, b Long) (GraphNode, error) {
	return returnHelper(C.fadd_cl(a.ref, C.long(b)))
}

func AddGraphFloat(a GraphNode, b Float) (GraphNode, error) {
	return returnHelper(C.fadd_cf(a.ref, C.float(b)))
}

func AddGraphDouble(a GraphNode, b Double) (GraphNode, error) {
	return returnHelper(C.fadd_cd(a.ref, C.double(b)))
}

/////////////////

func SubGraphGraph(a GraphNode, b GraphNode) (GraphNode, error) {
	return returnHelper(C.fsub_g(a.ref, b.ref))
}

func SubGraphInt(a GraphNode, b Int) (GraphNode, error) {
	return returnHelper(C.fsub_ci(a.ref, C.int(b)))
}

func SubGraphLong(a GraphNode, b Long) (GraphNode, error) {
	return returnHelper(C.fsub_cl(a.ref, C.long(b)))
}

func SubGraphFloat(a GraphNode, b Float) (GraphNode, error) {
	return returnHelper(C.fsub_cf(a.ref, C.float(b)))
}

func SubGraphDouble(a GraphNode, b Double) (GraphNode, error) {
	return returnHelper(C.fsub_cd(a.ref, C.double(b)))
}

func SubIntGraph(b Int, a GraphNode) (GraphNode, error) {
	return returnHelper(C.fsub_ci(C.int(b), a.ref))
}

func SubLongGraph(b Long, a GraphNode) (GraphNode, error) {
	return returnHelper(C.fsub_cl(C.long(b), a.ref))
}

func SubFloatGraph(b Float, a GraphNode) (GraphNode, error) {
	return returnHelper(C.fsub_cf(C.float(b), a.ref))
}

func SubDoubleGraph(b Double, a GraphNode) (GraphNode, error) {
	return returnHelper(C.fsub_cd(C.double(b), a.ref))
}

//////////////////

func DivGraphGraph(a GraphNode, b GraphNode) (GraphNode, error) {
	return returnHelper(C.fdiv_g(a.ref, b.ref))
}

func DivGraphInt(a GraphNode, b Int) (GraphNode, error) {
	return returnHelper(C.fdiv_ci(a.ref, C.int(b)))
}

func DivGraphLong(a GraphNode, b Long) (GraphNode, error) {
	return returnHelper(C.fdiv_cl(a.ref, C.long(b)))
}

func DivGraphFloat(a GraphNode, b Float) (GraphNode, error) {
	return returnHelper(C.fdiv_cf(a.ref, C.float(b)))
}

func DivGraphDouble(a GraphNode, b Double) (GraphNode, error) {
	return returnHelper(C.fdiv_cd(a.ref, C.double(b)))
}

func DivIntGraph(b Int, a GraphNode) (GraphNode, error) {
	return returnHelper(C.fdiv_ci(C.int(b), a.ref))
}

func DivLongGraph(b Long, a GraphNode) (GraphNode, error) {
	return returnHelper(C.fdiv_cl(C.long(b), a.ref))
}

func DivFloatGraph(b Float, a GraphNode) (GraphNode, error) {
	return returnHelper(C.fdiv_cf(C.float(b), a.ref))
}

func DivDoubleGraph(b Double, a GraphNode) (GraphNode, error) {
	return returnHelper(C.fdiv_cd(C.double(b), a.ref))
}

//////////////

func MulGraphGraph(a GraphNode, b GraphNode) (GraphNode, error) {
	return returnHelper(C.fmul_g(a.ref, b.ref))
}

func MulGraphInt(a GraphNode, b Int) (GraphNode, error) {
	return returnHelper(C.fmul_ci(a.ref, C.int(b)))
}

func MulGraphLong(a GraphNode, b Long) (GraphNode, error) {
	return returnHelper(C.fmul_cl(a.ref, C.long(b)))
}

func MulGraphFloat(a GraphNode, b Float) (GraphNode, error) {
	return returnHelper(C.fmul_cf(a.ref, C.float(b)))
}

func MulGraphDouble(a GraphNode, b Double) (GraphNode, error) {
	return returnHelper(C.fmul_cd(a.ref, C.double(b)))
}

///////////////

func PowGraphGraph(a GraphNode, b GraphNode) (GraphNode, error) {
	return returnHelper(C.fpow_g(a.ref, b.ref))
}

func PowGraphInt(a GraphNode, b Int) (GraphNode, error) {
	return returnHelper(C.fpow_ci(a.ref, C.int(b)))
}

func PowGraphLong(a GraphNode, b Long) (GraphNode, error) {
	return returnHelper(C.fpow_cl(a.ref, C.long(b)))
}

func PowGraphFloat(a GraphNode, b Float) (GraphNode, error) {
	return returnHelper(C.fpow_cf(a.ref, C.float(b)))
}

func PowGraphDouble(a GraphNode, b Double) (GraphNode, error) {
	return returnHelper(C.fpow_cd(a.ref, C.double(b)))
}

////////////////

// Log takes the element wise logarithm naturalis of x.
func Log(x GraphNode) (GraphNode, error) {
	return returnHelper(C.flog(x.ref))
}

// Log2 takes the element wise base 10 logarithm of x.
func Log2(x GraphNode) (GraphNode, error) {
	return returnHelper(C.flog2(x.ref))
}

// Log10 takes the element wise base 10 logarithm of x.
func Log10(x GraphNode) (GraphNode, error) {
	return returnHelper(C.flog10(x.ref))
}

// Sin takes the element wise sinus of x.
func Sin(x GraphNode) (GraphNode, error) {
	return returnHelper(C.fsin(x.ref))
}

// Sqrt takes the element wise square root of x.
func Sqrt(x GraphNode) (GraphNode, error) {
	return returnHelper(C.fsqrt_g(x.ref))
}

// Exp takes each element as the exponent for power of base e.
func Exp(x GraphNode) (GraphNode, error) {
	return returnHelper(C.fexp(x.ref))
}

// Cos takes the element wise cosine of x.
func Cos(x GraphNode) (GraphNode, error) {
	return returnHelper(C.fcos(x.ref))
}

// Tan takes the element wise tangent of x.
func Tan(x GraphNode) (GraphNode, error) {
	return returnHelper(C.ftan(x.ref))
}

// Asin takes the element wise inverse sinus of x.
func Asin(x GraphNode) (GraphNode, error) {
	return returnHelper(C.fasin(x.ref))
}

// Acos takes the element wise inverse cosine of x.
func Acos(x GraphNode) (GraphNode, error) {
	return returnHelper(C.facos(x.ref))
}

// Atan takes the element wise inverse tangent of x.
func Atan(x GraphNode) (GraphNode, error) {
	return returnHelper(C.fatan(x.ref))
}

// Neg swaps the sign of each element.
func Neg(x GraphNode) (GraphNode, error) {
	return returnHelper(C.fneg(x.ref))
}

/*
Sign applies the sign function to each element.
i.e. x[i] = 1 if x[i] >= 0 else x[i] = -1
The input flint [x] must have an integer type.
This function returns a [F_INT32] [GraphNode].
*/
func Sign(x GraphNode) (GraphNode, error) {
	return returnHelper(C.fsign(x.ref))
}

/*
Even gives the result of module 2 for each element.
i.e. x[i] = 1 if x[i] mod 2 == 0 else x[i] = 0
This function returns a [F_INT32] [GraphNode].
*/
func Even(x GraphNode) (GraphNode, error) {
	return returnHelper(C.feven(x.ref))
}

/////////////////

func LessGraphGraph(a GraphNode, b GraphNode) (GraphNode, error) {
	return returnHelper(C.fless_g(a.ref, b.ref))
}

func LessGraphInt(a GraphNode, b Int) (GraphNode, error) {
	return returnHelper(C.fless_ci(a.ref, C.int(b)))
}

func LessGraphLong(a GraphNode, b Long) (GraphNode, error) {
	return returnHelper(C.fless_cl(a.ref, C.long(b)))
}

func LessGraphFloat(a GraphNode, b Float) (GraphNode, error) {
	return returnHelper(C.fless_cf(a.ref, C.float(b)))
}

func LessGraphDouble(a GraphNode, b Double) (GraphNode, error) {
	return returnHelper(C.fless_cd(a.ref, C.double(b)))
}

/////////////////

func GreaterGraphGraph(a GraphNode, b GraphNode) (GraphNode, error) {
	return returnHelper(C.fgreater_g(a.ref, b.ref))
}

func GreaterGraphInt(a GraphNode, b Int) (GraphNode, error) {
	return returnHelper(C.fgreater_ci(a.ref, C.int(b)))
}

func GreaterGraphLong(a GraphNode, b Long) (GraphNode, error) {
	return returnHelper(C.fgreater_cl(a.ref, C.long(b)))
}

func GreaterGraphFloat(a GraphNode, b Float) (GraphNode, error) {
	return returnHelper(C.fgreater_cf(a.ref, C.float(b)))
}

func GreaterGraphDouble(a GraphNode, b Double) (GraphNode, error) {
	return returnHelper(C.fgreater_cd(a.ref, C.double(b)))
}

/////////////////

func EqualGraphGraph(a GraphNode, b GraphNode) (GraphNode, error) {
	return returnHelper(C.fequal_g(a.ref, b.ref))
}

func EqualGraphInt(a GraphNode, b Int) (GraphNode, error) {
	return returnHelper(C.fequal_ci(a.ref, C.int(b)))
}

func EqualGraphLong(a GraphNode, b Long) (GraphNode, error) {
	return returnHelper(C.fequal_cl(a.ref, C.long(b)))
}

func EqualGraphFloat(a GraphNode, b Float) (GraphNode, error) {
	return returnHelper(C.fequal_cf(a.ref, C.float(b)))
}

func EqualGraphDouble(a GraphNode, b Double) (GraphNode, error) {
	return returnHelper(C.fequal_cd(a.ref, C.double(b)))
}

////////////////

/*
Matmul carries out matrix multiplication on the last two dimensions of the tensors.

E.g. a matrix multiplication of two tensors with shapes (64, 32, 16) and (16, 24) will yield a flint with shape (64, 32, 24).
Since for one entry of the flint multiple other previous entries are needed, the operand tensors need to be executed first.
Therefor the method will implicitly (or eagerly) execute the two parameter nodes [a] and [b] if their data is not already present.
*/
func Matmul(a GraphNode, b GraphNode) (GraphNode, error) {
	return returnHelper(C.fmatmul(a.ref, b.ref))
}

/*
Flatten the complete flint to a flint with one dimension.
E.g:

	Flatten([[[3, 1, 4], [2, 1, 5]], [[0, 4, 2], [4, 7, 9]]]) =
		[3, 1, 4, 2, 1, 5, 0, 4, 2, 4, 7, 9].
*/
func Flatten(a GraphNode) (GraphNode, error) {
	return returnHelper(C.fflatten(a.ref))
}

/*
FlattenDim flattens a flint [a] with n dimensions along dimension [dim], resulting in a flint with n-1 dimensions.
Flattening a dimension will remove it from the shape of the flint, therefor it's not possible to flatten the dimension 0.

E.g:

	FlattenDim([[[3, 1, 4], [2, 1, 5]], [[0, 4, 2], [4, 7, 9]]], 1) =
		[[3,1,4], [2,1,5], [0,4,2], [4,7,9]]
*/
func FlattenDim(a GraphNode, dim int) (GraphNode, error) {
	return returnHelper(C.fflatten_dimension(a.ref, C.int(dim)))
}

/*
Convert the data of [a] to the type given by [newType].
*/
func Convert(a GraphNode, newType DataType) (GraphNode, error) {
	return returnHelper(C.fconvert(a.ref, C.enum_FType(newType)))
}

/*
Reshape the underlying data of the flint to the new shape.
The product of each dimension of the new shape must be the same as the product of the dimensions of the previous shape.
This means it must describe the same number of entries of the flint.
*/
func Reshape(a GraphNode, shape Shape) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)
	return returnHelper(C.freshape(a.ref, &(newShape[0]), C.int(len(shape))))
}

////////////////////

func MinimumGraphGraph(a GraphNode, b GraphNode) (GraphNode, error) {
	return returnHelper(C.fmin_g(a.ref, b.ref))
}

func MinimumGraphInt(a GraphNode, b Int) (GraphNode, error) {
	return returnHelper(C.fmin_ci(a.ref, C.int(b)))
}

func MinimumGraphLong(a GraphNode, b Long) (GraphNode, error) {
	return returnHelper(C.fmin_cl(a.ref, C.long(b)))
}

func MinimumGraphFloat(a GraphNode, b Float) (GraphNode, error) {
	return returnHelper(C.fmin_cf(a.ref, C.float(b)))
}

func MinimumGraphDouble(a GraphNode, b Double) (GraphNode, error) {
	return returnHelper(C.fmin_cd(a.ref, C.double(b)))
}

///////////////////

func MaximumGraphGraph(a GraphNode, b GraphNode) (GraphNode, error) {
	return returnHelper(C.fmax_g(a.ref, b.ref))
}

func MaximumGraphInt(a GraphNode, b Int) (GraphNode, error) {
	return returnHelper(C.fmax_ci(a.ref, C.int(b)))
}

func MaximumGraphLong(a GraphNode, b Long) (GraphNode, error) {
	return returnHelper(C.fmax_cl(a.ref, C.long(b)))
}

func MaximumGraphFloat(a GraphNode, b Float) (GraphNode, error) {
	return returnHelper(C.fmax_cf(a.ref, C.float(b)))
}

func MaximumGraphDouble(a GraphNode, b Double) (GraphNode, error) {
	return returnHelper(C.fmax_cd(a.ref, C.double(b)))
}

///////////////////

/*
ReduceSum reduces one dimension of the flint by additive folding.

E.g:

	ReduceSum([[1,2,3], [4,5,6]], 0) = [5,7,9]
	ReduceSum([[1,2,3], [4,5,6]], 1) = [6,15]

The results of the predecessor node must be available, to
ensure that the method may execute the parameter node.
*/
func ReduceSum(a GraphNode, dim int) (GraphNode, error) {
	return returnHelper(C.freduce_sum(a.ref, C.int(dim)))
}

/*
ReduceMul reduces one dimension of the flint by multiplicative folding.

E.g:

	ReduceMul([[1,2,3], [4,5,6]], 0) = [4,10,18]
	ReduceMul([[1,2,3], [4,5,6]], 1) = [6, 120]

The results of the predecessor node must be available; to ensure that the method may execute the parameter node.
*/
func ReduceMul(a GraphNode, dim int) (GraphNode, error) {
	return returnHelper(C.freduce_mul(a.ref, C.int(dim)))
}

/*
ReduceMin reduces one dimension of the flint by keeping the minimum.

E.g:

	ReduceMin([[1,32,3], [4,5,3]], 0) = [1,5,3]
	ReduceMin([[9,2,3], [-1,5,6]], 1) = [2, -1]

The results of the predecessor node must be available; to ensure that the method may execute the parameter node.
*/
func ReduceMin(a GraphNode, dim int) (GraphNode, error) {
	return returnHelper(C.freduce_min(a.ref, C.int(dim)))
}

/*
ReduceMax reduces one dimension of the flint by keeping the maximum.

E.g:

	ReduceMax([[1,32,3], [4,5,3]], 0) = [4,32,3]
	ReduceMax([[9,2,3], [-1,5,6]], 1) = [9, 6]

The results of the predecessor node must be available; to ensure that the method may execute the parameter node.
*/
func ReduceMax(a GraphNode, dim int) (GraphNode, error) {
	return returnHelper(C.freduce_max(a.ref, C.int(dim)))
}

/*
Slice selects a slice of the flint with a dimension wise start and end index.
[start] and [end] are arrays with as many entries as the flint has dimensions.
They may contain negative values, which are then subtracted from the end of the flint
(e.g. -1 means the element before the last element).
[start] is inclusive and describes the start index of the selection per dimension and [end] describes the end index per dimension and is exclusive.
*/
func Slice(a GraphNode, start Axes, end Axes) (GraphNode, error) {
	newStart := convertArray[int, C.long](start)
	newEnd := convertArray[int, C.long](end)
	return returnHelper(C.fslice(a.ref, &(newStart[0]), &(newEnd[0])))
}

/*
SliceWithStride selects a slice of the flint [node] with a dimension wise start index, end index and stride.
[start], [end] and [stride] are arrays with as many entries as the flint has dimensions.
[start] and [end] may contain negative values, which are then subtracted from the end of the flint
(e.g. -1 means the element before the last element).
[start] is inclusive and describes the start index of the selection per dimension and [end] describes the end index per dimension and is exclusive.
[stride] contains the per dimension step size (e.g. 2 meaning every second element will be selected etc.) and may be negative as well,
which reverses the traversal order (the first elements are selected as the last ones).
For a negative stride, [start] > [end] must hold (for a positive of course [end] > [start]) for each dimension.
*/
func SliceWithStride(node GraphNode, start Axes, end Axes, stride Stride) (GraphNode, error) {
	// TODO: check that axes has the right length compared to node! (or does the C function do this?)
	newStart := convertArray[int, C.long](start)
	newEnd := convertArray[int, C.long](end)
	newStride := convertArray[int, C.long](stride)

	return returnHelper(C.fslice_step(node.ref, &(newStart[0]), &(newEnd[0]), &(newStride[0])))
}

/*
Extend creates a new flint of zeroes with the requested shape.
The original flint is embedded at the given indices.
  - [node]: original flint which shape is to be extended
  - [shape]: array of new sizes per dimension. Has the same number of entries and the shape of [node].
  - [insertAt]: array with indices per dimension of [node], denoting where [node] is to be placed in the resulting flint
*/
func Extend(node GraphNode, shape Shape, insertAt Axes) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)
	newInsertAt := convertArray[int, C.size_t](insertAt)
	return returnHelper(C.fextend(node.ref, &(newShape[0]), &(newInsertAt[0])))
}

/*
ExtendWithStride creates a new flint of zeroes with the requested shape.
The original flint is embedded at the given indices.
  - [node]: original flint which shape is to be extended,
  - [shape]: array of new sizes per dimension. Has the same number of entries as [node] has dimensions.
  - [insertAt]: array with indices per dimension denoting where [node] is to be placed in the resulting flint. Has a value per dimension of [node].
  - [stride]: allows to pull apart [node], em-placing zeros between each value of [node]. Has a value per dimension.
*/
func ExtendWithStride(node GraphNode, shape Shape, insertAt Axes, stride Stride) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)
	newInsertAt := convertArray[int, C.size_t](insertAt)
	newStride := convertArray[int, C.long](stride)

	return returnHelper(C.fextend_step(node.ref, &(newShape[0]), &(newInsertAt[0]), &(newStride[0])))
}

/*
Concat two nodes ([nodeA], [nodeB]) with each other along an [axis].
The nodes have to have the same type and dimensions.

E.g:

	Concat({[[0, 1], [2, 3]], [[4, 5], [6, 7]]}, 0) = [[0, 1], [2, 3], [4, 5], [6, 7]]

	Concat({[[0, 1], [2, 3]], [[4, 5], [6, 7]]}, 1) = [[0, 1, 4, 5], [2, 3, 6, 7]]
*/
func Concat(nodeA GraphNode, nodeB GraphNode, axis uint) (GraphNode, error) {
	return returnHelper(C.fconcat(nodeA.ref, nodeB.ref, C.uint(axis)))
}

/*
Expand adds a new dimension at an arbitrary position to the flint and repeats the following dimensions to match a given shape.
  - [axis]: the dimension prior to which the new dimension will be inserted (0 means a new dimension in the front, n means as a new last dimension).
  - [size]: the new size of that dimension (repeats the following dimensions ax_size - 1 times).
*/
func Expand(a GraphNode, axis uint, size uint) (GraphNode, error) {
	return returnHelper(C.fexpand(a.ref, C.uint(axis), C.uint(size)))
}

/*
Abs takes the elementwise absolute value of [node], i.e. |a[i]|
*/
func Abs(node GraphNode) (GraphNode, error) {
	return returnHelper(C.fabs_g(node.ref))
}

/*
Repeat dimensions of a flint multiple times.
  - [node]: the node in which dimensions are to be repeated
  - [axes]: array with the same number of entries as the flint has dimensions

E.g:

	Repeat([[0,1], [2,3]], [2, 3]) =
		[[0,1,0,1,0,1], [2,3,2,3,2,3], [0,1,0,1,0,1], [2,3,2,3,2,3]]
*/
func Repeat(node GraphNode, repetitions Axes) (GraphNode, error) {
	newRepetitions := convertArray[int, C.int](repetitions)
	return returnHelper(C.frepeat(node.ref, &(newRepetitions[0])))
}

/*
Transpose flint [node] along multiple dimensions.
The array [axes] has the same number of entries as [node] has dimensions, which gives the permutations of dimensions.

The flint will have a resulting shape in which the size each dimension corresponds to the former size in dimension in [axes].
*/
func Transpose(node GraphNode, axes Axes) (GraphNode, error) {
	newAxes := convertArray[int, C.int](axes)
	return returnHelper(C.ftranspose(node.ref, &(newAxes[0])))
}

/*
Convolve the n-dimensional input flint [node] with an n-dimensional filter
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
	return returnHelper(C.fconvolve(node.ref, kernel.ref, &(newStride[0])))
}

/*
Index selects single elements with an index-flint (integer flint containing indices for the selected dimension).
It indexes a dimension of the input flint and the result has the shape of the input flint except for the indexed dimension.
It is assumed that except for the last entry its shape is a prefix of the shape of the input flint and the indexing will occur in the matched subsets.
The last dimension of indices is the one indexed in node.

Example 1: Take the "subset" of the matrix where the first two slices are swapped:

	Index([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
			[1, 0]) =
		[[[4, 5], [6, 7]], [[0, 1], [2, 3]]]

Example 2:

	Index([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
			[0, 0, 1]) =
		[[[0, 1], [0, 1], [1, 2]], [[3, 4], [3, 4], [5, 6]], [[7, 8], [7, 8], [9, 10]]]

Example 3:

	Index([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
			[[0], [1], [0]]) =
		[[[0], [2]], [[5], [7]], [[8], [10]]]

Example 4:

	Index([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
			[[0, 0], [1, 0], [0, 1]]) =
		[[[0, 0], [2, 2]], [[5, 4], [7, 6]], [[8, 9], [10, 11]]]
*/
func Index(node GraphNode, indices GraphNode) (GraphNode, error) {
	return returnHelper(C.findex(node.ref, indices.ref))
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
	return returnHelper(C.findex_set(nodeA.ref, nodeB.ref, indices.ref))
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

	return returnHelper(C.fsliding_window(node.ref, &(newSize[0]), &(newStride[0])))
}

// TODO: comment from flint.h
func UnslideWindow(node GraphNode, size Shape, stride Stride) (GraphNode, error) {
	newSize := convertArray[uint, C.size_t](size)
	newStride := convertArray[int, C.uint](stride)
	return returnHelper(C.funslide_window(node.ref, &(newSize[0]), &(newStride[0])))
}

/*
Permute randomly permutes (= swaps multiple elements with each other without creating, copying or deleting new ones) one axis of the input flint.
*/
func Permute(a GraphNode, axis uint) (GraphNode, error) {
	return returnHelper(C.fpermutate(a.ref, C.uint(axis)))
}

func PoolingSum(node GraphNode, size Shape, stride Stride) (GraphNode, error) {
	newSize := convertArray[uint, C.size_t](size)
	newStride := convertArray[int, C.uint](stride)
	return returnHelper(C.fpooling_sum(node.ref, &(newSize[0]), &(newStride[0])))
}

func PoolingMax(node GraphNode, size Shape, stride Stride) (GraphNode, error) {
	newSize := convertArray[uint, C.size_t](size)
	newStride := convertArray[int, C.uint](stride)
	return returnHelper(C.fpooling_max(node.ref, &(newSize[0]), &(newStride[0])))
}

func Dropout(node GraphNode, probability Double) (GraphNode, error) {
	return returnHelper(C.fdropout(node.ref, C.double(probability)))
}
