// Package flint contains all types and function declarations similar to [../flint.h]
package flint

/*
#cgo LDFLAGS: -lflint -lOpenCL -lstdc++
#include "../flint.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

///////////////
/// SETUP
//////////////

type Backend int

const (
	BACKEND_ONLY_CPU Backend = iota
	BACKEND_ONLY_GPU
	BACKEND_BOTH
)

func Init(backend Backend) {
	C.flintInit(C.int(backend))
}

func InitializedBackend() Backend {
	// FIXME: this might break when a new backend is added,
	// as this go frontend does not doe the bit magic
	return Backend(C.flintInitializedBackends())
}

func Cleanup() {
	C.flintCleanup()
}

///////////////
/// Logging
//////////////

type loggingLevel int

const (
	NO_LOGGING loggingLevel = iota
	ERROR
	WARNING
	INFO
	VERBOSE
	DEBUG
)

func SetLoggingLevel(level loggingLevel) {
	C.fSetLoggingLevel(C.int(level))
}

func Log(level loggingLevel, message string) {
	unsafeMessage := C.CString(message)
	defer C.free(unsafe.Pointer(unsafeMessage))
	C.flogging(C.enum_FLogType(level), unsafeMessage)
}

///////////////
/// Images
//////////////

type imageFormat int

const (
	PNG imageFormat = iota
	JPEG
	BMP
)

func SetEagerExecution(turnOn bool) {
	if turnOn == true {
		C.fEnableEagerExecution()
	} else {
		C.fDisableEagerExecution()
	}
}

func IsEagerExecution() bool {
	res := int(C.fIsEagerExecution())
	if res == 0 {
		return false
	} else {
		return true
	}
}

// /////////////
// / Creating and Executing Tensors
// ////////////

type Shape []uint64

type tensorDataType interface {
	int32 | int64 | float32 | float64
}

type Tensor[T tensorDataType] struct {
	node GraphNode[T]
}
type FloatTensor Tensor[float32]
type IntTensor Tensor[int32]
type DoubleTensor Tensor[float64]
type LongTensor Tensor[int64]

type GraphNode[T tensorDataType] struct {
	ref   *C.FGraphNode
	shape Shape
}

type Result struct {
	shape Shape
	data  any
}

func getFlintType(obj any) C.enum_FType {
	switch obj.(type) {
	case int32, int:
		return C.F_INT32
	case int64:
		return C.F_INT64
	case float32:
		return C.F_FLOAT32
	case float64:
		return C.F_FLOAT64
	default:
		panic("invalid data type")
	}
}

func CreateGraph[T tensorDataType](data []T, shape Shape) GraphNode[T] {
	//datatype := getFlintType(T(data))
	//datatype := C.F_FLOAT32
	shapePtr := (*C.size_t)(unsafe.Pointer(&shape[0]))
	dataPtr := unsafe.Pointer(&data[0])
	flintNode := C.fCreateGraph(dataPtr, C.int(len(data)), 2, shapePtr, C.int(len(shape)))
	return GraphNode[T]{
		ref:   flintNode,
		shape: shape,
	}
}

func CreateGraphConstant[T tensorDataType](value T, shape Shape) GraphNode[T] {
	shapePtr := (*C.size_t)(unsafe.Pointer(&shape[0]))
	var flintNode *C.FGraphNode
	dimensions := C.int(len(shape))
	switch any(value).(type) {
	case int32:
		flintNode = C.fconstant_i(C.int(value), shapePtr, dimensions)
	case int64:
		flintNode = C.fconstant_l(C.long(value), shapePtr, dimensions)
	case float32:
		flintNode = C.fconstant_f(C.float(value), shapePtr, dimensions)
	case float64:
		flintNode = C.fconstant_d(C.double(value), shapePtr, dimensions)
	default:
		panic("invalid data type")
	}
	return GraphNode[T]{
		ref:   flintNode,
		shape: shape,
	}
}

func (a GraphNode[T]) Result() Result {
	flintNode := C.fCalculateResult(a.ref)
	return Result{
		shape: *((*Shape)(unsafe.Pointer(&flintNode.operation.shape))),
		data:  flintNode.result_data.data,
	}
}

func (a GraphNode[T]) Serialize() string {
	ptr := C.fserialize(a.ref, nil)
	defer C.free(unsafe.Pointer(ptr))
	return C.GoString(ptr)
}

func Deserialize[T tensorDataType](data string) GraphNode[T] {
	unsafeData := C.CString(data)
	defer C.free(unsafe.Pointer(unsafeData))
	node := C.fdeserialize(unsafeData)
	return GraphNode[T]{ref: node}
}

///////////////
/// Tensor Operations
//////////////

func (a GraphNode[T]) Add(b GraphNode[T]) GraphNode[T] {
	flintNode := C.fadd_g(a.ref, b.ref)
	return GraphNode[T]{
		ref:   flintNode,
		shape: a.shape,
	}
}

type Node struct {
	ref          *C.FGraphNode // FIXME: or unsafe pointer?
	shape        Shape
	predecessors []Node
}

type Stride []int
type Axes []int // needs to have one entry for each dimension of tensor

func TestConstant() Node {
	var newShape = []C.ulong{C.ulong(2), C.ulong(2)}
	newShapePtr := (*C.ulong)(unsafe.Pointer(&newShape[0]))

	var n *C.FGraphNode = C.fconstant_i(C.int(5), newShapePtr, C.int(2))
	fmt.Println(n)

	return Node{ref: n}
}

func (a *Node) matmul(b *Node) Node {
	C.fmatmul(a.ref, b.ref)
	return Node{ref: nil}
}

func (a *Node) flatten() Node {
	C.fflatten(a.ref)
	return Node{ref: nil}
}

func (a *Node) flattenDim(dim int) Node {
	C.fflatten_dimension(a.ref, C.int(dim))
	return Node{ref: nil}
}

// FIXME: type param, or generic?
func (a *Node) convert() Node {
	C.fconvert(a.ref, C.F_INT32)
	return Node{ref: nil}
}

func (a *Node) reshape(shape Shape) Node {
	C.freshape(a.ref, nil, C.int(len(shape)))
	return Node{ref: nil}
}

func (a *Node) min(b any) Node {
	// FIXME: any has to be float, double, int, long or GraphNode type
	//switch any(b).(type) {
	//case *Node:
	//	C.fmin_g(a.ref, b.ref)
	//case int32, int:
	//	C.fmin_ci(a.ref, C.int(b))
	//case int64:
	//	C.fmin_cl(a.ref, C.long(b))
	//case float32:
	//	C.fmin_cf(a.ref, C.float(b))
	//case float64:
	//	C.fmin_cd(a.ref, C.double(b))
	//default:
	//	panic("invalid type")
	//}
	return Node{ref: nil}
}

func (a *Node) max(b any) Node {
	// FIXME: any has to be float, double, int, long or GraphNode type
	//switch b.(type) {
	//case *Node:
	//	C.fmax_g(a.ref, b.ref)
	//case int32, int:
	//	C.fmax_ci(a.ref, C.int(b))
	//case int64:
	//	C.fmax_cl(a.ref, C.long(b))
	//case float32:
	//	C.fmax_cf(a.ref, C.float(b))
	//case float64:
	//	C.fmax_cd(a.ref, C.double(b))
	//default:
	//	panic("invalid type")
	//}
	return Node{ref: nil}
}

func (a *Node) reduceSum(dim int) Node {
	C.freduce_sum(a.ref, C.int(dim))
	return Node{ref: nil}
}

func (a *Node) reduceMul(dim int) Node {
	C.freduce_mul(a.ref, C.int(dim))
	return Node{ref: nil}
}

func (a *Node) reduceMin(dim int) Node {
	C.freduce_min(a.ref, C.int(dim))
	return Node{ref: nil}
}

func (a *Node) reduceMax(dim int) Node {
	C.freduce_max(a.ref, C.int(dim))
	return Node{ref: nil}
}

func (a *Node) slice(start Axes, end Axes) Node {
	C.fslice(a.ref, nil, nil)
	return Node{ref: nil}
}

func (a *Node) sliceWithStride(start Axes, end Axes, stride Stride) Node {
	C.fslice_step(a.ref, nil, nil, nil)
	return Node{ref: nil}
}

func (a *Node) extend(shape Shape, insertAt Axes) Node {
	C.fextend(a.ref, nil, nil)
	return Node{ref: nil}
}

func (a *Node) extendWithStride(shape Shape, insertAt Axes, stride Stride) Node {
	C.fextend_step(a.ref, nil, nil, nil)
	return Node{ref: nil}
}

func (a *Node) concat(b *Node, axis uint) Node {
	C.fconcat(a.ref, b.ref, C.uint(axis))
	return Node{ref: nil}
}

func (a *Node) abs() Node {
	C.fabs_g(a.ref)
	return Node{ref: nil}
}

func (a *Node) repeat(repetitions Axes) Node {
	C.frepeat(a.ref, nil)
	return Node{ref: nil}
}

func (a *Node) transpose(axes Axes) Node {
	C.ftranspose(a.ref, nil)
	return Node{ref: nil}
}

func (a *Node) convolve(kernel *Node, stride Stride) Node {
	C.fconvolve(a.ref, kernel.ref, nil)
	return Node{ref: nil}
}

func (a *Node) slide(kernel *Node, stride Stride) Node {
	C.fslide(a.ref, kernel.ref, nil)
	return Node{ref: nil}
}

func (a *Node) index(indices *Node) Node {
	C.findex(a.ref, indices.ref)
	return Node{ref: nil}
}

func (a *Node) indexSet(b *Node, indices *Node) Node {
	C.findex_set(a.ref, b.ref, indices.ref)
	return Node{ref: nil}
}

func (a *Node) slidingWindow(size Stride, stride Stride) Node {
	C.fsliding_window(a.ref, nil, nil)
	return Node{ref: nil}
}

func (a *Node) permute(axis uint) Node {
	C.fpermutate(a.ref, C.uint(axis))
	return Node{ref: nil}
}
