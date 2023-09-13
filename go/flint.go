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
// Types and Structs
//////////////

type Tensor[T Numeric] struct {
	GraphNode GraphNode
}
type FloatTensor Tensor[float32]
type IntTensor Tensor[int32]
type DoubleTensor Tensor[float64]
type LongTensor Tensor[int64]

type GraphNode struct {
	ref *C.FGraphNode
}

type Result struct {
	shape Shape
	data  any
}

type tensorDataType interface {
	~int32 | ~int64 | ~float32 | ~float64
}
type Numeric tensorDataType

type Stride []int
type Axes []int // needs to have one entry for each dimension of tensor
type Shape []uint64

//////////////
// SETUP
/////////////

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

//////////////
// Logging
/////////////

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

// /////////////
// Creating and Executing Tensors
// ////////////

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

func CreateGraph[T Numeric](data []T, shape Shape) GraphNode {
	//datatype := getFlintType(T(data))
	//datatype := C.F_FLOAT32
	shapePtr := (*C.size_t)(unsafe.Pointer(&shape[0]))
	dataPtr := unsafe.Pointer(&data[0])
	flintNode := C.fCreateGraph(dataPtr, C.int(len(data)), 2, shapePtr, C.int(len(shape)))
	return GraphNode{
		ref: flintNode,
	}
}

func CreateGraphConstant[T Numeric](value T, shape Shape) GraphNode {
	shapePtr := (*C.size_t)(unsafe.Pointer(&shape[0]))
	var flintNode *C.FGraphNode
	dimensions := C.int(len(shape))
	switch v := any(value).(type) {
	case int32:
		flintNode = C.fconstant_i(C.int(v), shapePtr, dimensions)
	case int64:
		flintNode = C.fconstant_l(C.long(v), shapePtr, dimensions)
	case float32:
		flintNode = C.fconstant_f(C.float(v), shapePtr, dimensions)
	case float64:
		flintNode = C.fconstant_d(C.double(v), shapePtr, dimensions)
	default:
		panic("invalid data type")
	}
	return GraphNode{
		ref: flintNode,
	}
}

func cToShape(ptr *C.size_t) Shape {
	return nil
}

func (a GraphNode) Result() Result {
	flintNode := C.fCalculateResult(a.ref)
	return Result{
		//shape: cToShape(flintNode.operation.shape),
		data: flintNode.result_data.data,
	}
}

func (a GraphNode) Serialize() []byte {
	var size C.size_t
	ptr := C.fserialize(a.ref, &size)
	defer C.free(unsafe.Pointer(ptr))
	return C.GoBytes(unsafe.Pointer(ptr), C.int(size))
}

func Deserialize(data []byte) GraphNode {
	unsafeData := C.CBytes(data)
	defer C.free(unsafe.Pointer(unsafeData))
	flintNode := C.fdeserialize((*C.char)(unsafeData))
	return GraphNode{ref: flintNode}
}

///////////////
/// Tensor Operations
//////////////

func describe(i any) {
	fmt.Printf("describe (value, underlying type): (%v, %T)\n", i, i)
}

func Add[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode

	switch c := any(b).(type) {
	case int32:
		flintNode = C.fadd_ci(a.ref, C.int(c))
	case int64:
		flintNode = C.fadd_cl(a.ref, C.long(c))
	case float32:
		flintNode = C.fadd_cf(a.ref, C.float(c))
	case float64:
		flintNode = C.fadd_cd(a.ref, C.double(c))
	case GraphNode:
		flintNode = C.fadd_g(a.ref, c.ref)
	default:
		panic("invalid type")
	}

	return GraphNode{
		ref: flintNode,
	}
}

// TODO: pow, mul, div

// TODO: trig functions

// TODO: neg

// TODO: even

func Less(a GraphNode, compareTo any) GraphNode {
	return GraphNode{}
}

func Greater(a GraphNode, compareTo any) GraphNode {
	return GraphNode{}
}

func Equal(a GraphNode, compareTo any) GraphNode {
	//switch compareTo.(type) {
	//case GraphNode:
	//	flintNode := C.fgreater_g(a.ref, compareTo.ref)
	//}
	return GraphNode{}
}

func Matmul(a GraphNode, b GraphNode) GraphNode {
	flintNode := C.fmatmul(a.ref, b.ref)
	return GraphNode{ref: flintNode}
}

func Flatten(a GraphNode) GraphNode {
	flintNode := C.fflatten(a.ref)
	return GraphNode{ref: flintNode}
}

func FlattenDim(a GraphNode, dim int) GraphNode {
	flintNode := C.fflatten_dimension(a.ref, C.int(dim))
	return GraphNode{ref: flintNode}
}

func Convert[T tensorDataType](a GraphNode, newType T) GraphNode {
	//newType := uint32(C.F_INT32)
	flintNode := C.fconvert(a.ref, newType)
	return GraphNode{ref: flintNode}
}

func Reshape(a GraphNode, shape Shape) GraphNode {
	flintNode := C.freshape(a.ref, nil, C.int(len(shape)))
	return GraphNode{ref: flintNode}
}

func min(a GraphNode, b any) GraphNode {
	// FIXME: any has to be float, double, int, long or GraphNode type
	//switch any(b).(type) {
	//case GraphNode:
	//	flintNode := C.fmin_g(a.ref, b.ref)
	//case int32, int:
	//	flintNode := C.fmin_ci(a.ref, C.int(b))
	//case int64:
	//	flintNode := C.fmin_cl(a.ref, C.long(b))
	//case float32:
	//	flintNode := C.fmin_cf(a.ref, C.float(b))
	//case float64:
	//	flintNode := C.fmin_cd(a.ref, C.double(b))
	//default:
	//	panic("invalid type")
	//}
	return GraphNode{ref: flintNode}
}

func max(a GraphNode, b any) GraphNode {
	// FIXME: any has to be float, double, int, long or GraphNode type
	//switch b.(type) {
	//case GraphNode:
	//	flintNode := C.fmax_g(a.ref, b.ref)
	//case int32, int:
	//	flintNode := C.fmax_ci(a.ref, C.int(b))
	//case int64:
	//	flintNode := C.fmax_cl(a.ref, C.long(b))
	//case float32:
	//	flintNode := C.fmax_cf(a.ref, C.float(b))
	//case float64:
	//	flintNode := C.fmax_cd(a.ref, C.double(b))
	//default:
	//	panic("invalid type")
	//}
	return GraphNode{ref: flintNode}
}

func ReduceSum(a GraphNode, dim int) GraphNode {
	flintNode := C.freduce_sum(a.ref, C.int(dim))
	return GraphNode{ref: flintNode}
}

func ReduceMul(a GraphNode, dim int) GraphNode {
	flintNode := C.freduce_mul(a.ref, C.int(dim))
	return GraphNode{ref: flintNode}
}

func ReduceMin(a GraphNode, dim int) GraphNode {
	flintNode := C.freduce_min(a.ref, C.int(dim))
	return GraphNode{ref: flintNode}
}

func ReduceMax(a GraphNode, dim int) GraphNode {
	flintNode := C.freduce_max(a.ref, C.int(dim))
	return GraphNode{ref: flintNode}
}

func Slice(a GraphNode, start Axes, end Axes) GraphNode {
	flintNode := C.fslice(a.ref, nil, nil)
	return GraphNode{ref: flintNode}
}

func SliceWithStride(a GraphNode, start Axes, end Axes, stride Stride) GraphNode {
	flintNode := C.fslice_step(a.ref, nil, nil, nil)
	return GraphNode{ref: flintNode}
}

func Extend(a GraphNode, shape Shape, insertAt Axes) GraphNode {
	flintNode := C.fextend(a.ref, nil, nil)
	return GraphNode{ref: flintNode}
}

func ExtendWithStride(a GraphNode, shape Shape, insertAt Axes, stride Stride) GraphNode {
	flintNode := C.fextend_step(a.ref, nil, nil, nil)
	return GraphNode{ref: flintNode}
}

func Concat(a GraphNode, b GraphNode, axis uint) GraphNode {
	flintNode := C.fconcat(a.ref, b.ref, C.uint(axis))
	return GraphNode{ref: flintNode}
}

func Expand(a GraphNode, axis uint, size uint) GraphNode {
	flintNode := C.fexpand(a.ref, C.uint(axis), C.uint(size))
	return GraphNode{ref: flintNode}
}

func Abs(a GraphNode) GraphNode {
	flintNode := C.fabs_g(a.ref)
	return GraphNode{ref: flintNode}
}

func Repeat(a GraphNode, repetitions Axes) GraphNode {
	flintNode := C.frepeat(a.ref, nil)
	return GraphNode{ref: flintNode}
}

func Transpose(a GraphNode, axes Axes) GraphNode {
	flintNode := C.ftranspose(a.ref, nil)
	return GraphNode{ref: flintNode}
}

func Convolve(a GraphNode, kernel GraphNode, stride Stride) GraphNode {
	flintNode := C.fconvolve(a.ref, kernel.ref, nil)
	return GraphNode{ref: flintNode}
}

func Slide(a GraphNode, kernel GraphNode, stride Stride) GraphNode {
	flintNode := C.fslide(a.ref, kernel.ref, nil)
	return GraphNode{ref: flintNode}
}

func Index(a GraphNode, indices GraphNode) GraphNode {
	flintNode := C.findex(a.ref, indices.ref)
	return GraphNode{ref: flintNode}
}

func IndexSet(a GraphNode, b GraphNode, indices GraphNode) GraphNode {
	flintNode := C.findex_set(a.ref, b.ref, indices.ref)
	return GraphNode{ref: flintNode}
}

func SlidingWindow(a GraphNode, size Stride, stride Stride) GraphNode {
	flintNode := C.fsliding_window(a.ref, nil, nil)
	return GraphNode{ref: flintNode}
}

func Permute(a GraphNode, axis uint) GraphNode {
	flintNode := C.fpermutate(a.ref, C.uint(axis))
	return GraphNode{ref: flintNode}
}
