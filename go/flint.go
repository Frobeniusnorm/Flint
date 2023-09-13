// Package flint contains all types and function declarations similar to [../flint.h]
package flint

/*
#cgo LDFLAGS: -lflint -lOpenCL -lstdc++
#include "../flint.h"
*/
import "C"
import (
	"fmt"
	"reflect"
	"unsafe"
)

///////////////
// Types and Structs
//////////////

type Tensor[T Numeric] struct {
	GraphNode graphNode
}
type FloatTensor Tensor[float32]
type IntTensor Tensor[int32]
type DoubleTensor Tensor[float64]
type LongTensor Tensor[int64]

type graphNode struct {
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

type Stride []int32 // needs to have one entry for each dimension of tensor
type Axes []int64   // needs to have one entry for each dimension of tensor
type Shape []uint64 // needs to have one entry for each dimension of tensor

func toC(obj any, outputType reflect.Type) any {
	//reflect.Value
	//return (outputType)(any)
	return nil
}

const (
	F_INT32 uint32 = iota
	F_INT64
	F_FLOAT32
	F_FLOAT64
)

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

//////////////
// Images
/////////////

type imageFormat int

const (
	PNG imageFormat = iota
	JPEG
	BMP
)

// /////////////
// Utility functions
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

func cToShape(ptr *C.size_t) Shape {
	return nil
}

func describe(i any) {
	fmt.Printf("describe (value, underlying type): (%v, %T)\n", i, i)
}

// /////////////
// Creating and Executing Tensors
// ////////////

func CreateGraph[T Numeric](data []T, shape Shape) graphNode {
	datatype := F_FLOAT32
	shapePtr := (*C.size_t)(unsafe.Pointer(&shape[0]))

	dataPtr := unsafe.Pointer(&data[0])
	flintNode := C.fCreateGraph(dataPtr, C.int(len(data)), datatype, shapePtr, C.int(len(shape)))
	return graphNode{
		ref: flintNode,
	}
}

func CreateGraphConstant[T Numeric](value T, shape Shape) graphNode {
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
	return graphNode{
		ref: flintNode,
	}
}

func (a graphNode) Result() Result {
	flintNode := C.fCalculateResult(a.ref)
	return Result{
		//shape: cToShape(flintNode.operation.shape),
		data: flintNode.result_data.data,
	}
}

func (a graphNode) Serialize() []byte {
	var size C.size_t
	ptr := C.fserialize(a.ref, &size)
	defer C.free(unsafe.Pointer(ptr))
	return C.GoBytes(unsafe.Pointer(ptr), C.int(size))
}

func Deserialize(data []byte) graphNode {
	unsafeData := C.CBytes(data)
	defer C.free(unsafe.Pointer(unsafeData))
	flintNode := C.fdeserialize((*C.char)(unsafeData))
	return graphNode{ref: flintNode}
}

///////////////
/// Tensor Operations
//////////////

func Add[T Numeric | graphNode](a graphNode, b T) graphNode {
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
	case graphNode:
		flintNode = C.fadd_g(a.ref, c.ref)
	default:
		panic("invalid type")
	}

	return graphNode{
		ref: flintNode,
	}
}

// TODO: pow, mul, div

// TODO: trig functions

// TODO: neg

// TODO: even

func Less(a graphNode, compareTo any) graphNode {
	return graphNode{}
}

func Greater(a graphNode, compareTo any) graphNode {
	return graphNode{}
}

func Equal(a graphNode, compareTo any) graphNode {
	//switch compareTo.(type) {
	//case graphNode:
	//	flintNode := C.fgreater_g(a.ref, compareTo.ref)
	//}
	return graphNode{}
}

func Matmul(a graphNode, b graphNode) graphNode {
	flintNode := C.fmatmul(a.ref, b.ref)
	return graphNode{ref: flintNode}
}

func Flatten(a graphNode) graphNode {
	flintNode := C.fflatten(a.ref)
	return graphNode{ref: flintNode}
}

func FlattenDim(a graphNode, dim int) graphNode {
	flintNode := C.fflatten_dimension(a.ref, C.int(dim))
	return graphNode{ref: flintNode}
}

func Convert[T tensorDataType](a graphNode, newType T) graphNode {
	newTypeXXXXX := uint32(C.F_INT32)
	flintNode := C.fconvert(a.ref, newTypeXXXXX)
	return graphNode{ref: flintNode}
}

func Reshape(a graphNode, shape Shape) graphNode {
	flintNode := C.freshape(a.ref, nil, C.int(len(shape)))
	return graphNode{ref: flintNode}
}

func min(a graphNode, b any) graphNode {
	// FIXME: any has to be float, double, int, long or graphNode type
	//switch any(b).(type) {
	//case graphNode:
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
	return graphNode{ref: nil}
}

func max(a graphNode, b any) graphNode {
	// FIXME: any has to be float, double, int, long or graphNode type
	//switch b.(type) {
	//case graphNode:
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
	return graphNode{ref: nil}
}

func ReduceSum(a graphNode, dim int) graphNode {
	flintNode := C.freduce_sum(a.ref, C.int(dim))
	return graphNode{ref: flintNode}
}

func ReduceMul(a graphNode, dim int) graphNode {
	flintNode := C.freduce_mul(a.ref, C.int(dim))
	return graphNode{ref: flintNode}
}

func ReduceMin(a graphNode, dim int) graphNode {
	flintNode := C.freduce_min(a.ref, C.int(dim))
	return graphNode{ref: flintNode}
}

func ReduceMax(a graphNode, dim int) graphNode {
	flintNode := C.freduce_max(a.ref, C.int(dim))
	return graphNode{ref: flintNode}
}

func Slice(a graphNode, start Axes, end Axes) graphNode {
	flintNode := C.fslice(a.ref, nil, nil)
	return graphNode{ref: flintNode}
}

func SliceWithStride(a graphNode, start Axes, end Axes, stride Stride) graphNode {
	flintNode := C.fslice_step(a.ref, nil, nil, nil)
	return graphNode{ref: flintNode}
}

func Extend(a graphNode, shape Shape, insertAt Axes) graphNode {
	flintNode := C.fextend(a.ref, nil, nil)
	return graphNode{ref: flintNode}
}

func ExtendWithStride(a graphNode, shape Shape, insertAt Axes, stride Stride) graphNode {
	flintNode := C.fextend_step(a.ref, nil, nil, nil)
	return graphNode{ref: flintNode}
}

func Concat(a graphNode, b graphNode, axis uint) graphNode {
	flintNode := C.fconcat(a.ref, b.ref, C.uint(axis))
	return graphNode{ref: flintNode}
}

func Expand(a graphNode, axis uint, size uint) graphNode {
	flintNode := C.fexpand(a.ref, C.uint(axis), C.uint(size))
	return graphNode{ref: flintNode}
}

func Abs(a graphNode) graphNode {
	flintNode := C.fabs_g(a.ref)
	return graphNode{ref: flintNode}
}

func Repeat(a graphNode, repetitions Axes) graphNode {
	flintNode := C.frepeat(a.ref, nil)
	return graphNode{ref: flintNode}
}

func Transpose(a graphNode, axes Axes) graphNode {
	flintNode := C.ftranspose(a.ref, nil)
	return graphNode{ref: flintNode}
}

func Convolve(a graphNode, kernel graphNode, stride Stride) graphNode {
	flintNode := C.fconvolve(a.ref, kernel.ref, nil)
	return graphNode{ref: flintNode}
}

func Slide(a graphNode, kernel graphNode, stride Stride) graphNode {
	flintNode := C.fslide(a.ref, kernel.ref, nil)
	return graphNode{ref: flintNode}
}

func Index(a graphNode, indices graphNode) graphNode {
	flintNode := C.findex(a.ref, indices.ref)
	return graphNode{ref: flintNode}
}

func IndexSet(a graphNode, b graphNode, indices graphNode) graphNode {
	flintNode := C.findex_set(a.ref, b.ref, indices.ref)
	return graphNode{ref: flintNode}
}

func SlidingWindow(a graphNode, size Shape, stride Stride) graphNode {
	flintNode := C.fsliding_window(a.ref, nil, nil)
	return graphNode{ref: flintNode}
}

func Permute(a graphNode, axis uint) graphNode {
	flintNode := C.fpermutate(a.ref, C.uint(axis))
	return graphNode{ref: flintNode}
}
