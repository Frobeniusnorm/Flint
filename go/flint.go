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

type Tensor[T tensorDataType] struct {
	GraphNode GraphNode
}
type FloatTensor Tensor[float32]
type IntTensor Tensor[int32]
type DoubleTensor Tensor[float64]
type LongTensor Tensor[int64]

type GraphNode struct {
	ref   *C.FGraphNode // FIXME: or unsafe pointer?
	shape Shape
}

type Result struct {
	shape Shape
	data  any
}

type tensorDataType interface {
	~int32 | ~int64 | ~float32 | ~float64
}

type Stride []int
type Axes []int // needs to have one entry for each dimension of tensor

type ValidType interface {
	RegisterType()
}

type (
	F_INT    int32
	F_LONG   int64
	F_FLOAT  float32
	F_DOUBLE float64
	F_GRAPH  GraphNode
)

func (x F_INT) RegisterType()    {}
func (x F_LONG) RegisterType()   {}
func (x F_FLOAT) RegisterType()  {}
func (x F_DOUBLE) RegisterType() {}
func (a F_GRAPH) RegisterType()  {}

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

func CreateGraph[T tensorDataType](data []T, shape Shape) GraphNode {
	//datatype := getFlintType(T(data))
	//datatype := C.F_FLOAT32
	shapePtr := (*C.size_t)(unsafe.Pointer(&shape[0]))
	dataPtr := unsafe.Pointer(&data[0])
	flintNode := C.fCreateGraph(dataPtr, C.int(len(data)), 2, shapePtr, C.int(len(shape)))
	return GraphNode{
		ref:   flintNode,
		shape: shape,
	}
}

func CreateGraphConstant[T tensorDataType](value T, shape Shape) GraphNode {
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
		ref:   flintNode,
		shape: shape,
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

func Deserialize[T tensorDataType](data []byte) GraphNode {
	unsafeData := C.CBytes(data)
	defer C.free(unsafe.Pointer(unsafeData))
	flintNode := C.fdeserialize((*C.char)(unsafeData))
	return GraphNode{ref: flintNode}
}

///////////////
/// Tensor Operations
//////////////

func describe(i ValidType) {
	fmt.Printf("describe (value, underlying type): (%v, %T)\n", i, i)
}

func Add2(a GraphNode, b ValidType) GraphNode {
	var flintNode *C.FGraphNode

	switch c := b.(type) {
	case F_INT:
		flintNode = C.fadd_ci(a.ref, C.int(c))
	case F_LONG:
		flintNode = C.fadd_cl(a.ref, C.long(c))
	case F_FLOAT:
		flintNode = C.fadd_cf(a.ref, C.float(c))
	case F_DOUBLE:
		flintNode = C.fadd_cd(a.ref, C.double(c))
	case F_GRAPH:
		flintNode = C.fadd_g(a.ref, c.ref)
	default:
		panic("invalid type")
	}

	return GraphNode{
		ref: flintNode,
	}
}

func Add[T tensorDataType | GraphNode](a GraphNode, b T) GraphNode {
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

func (a GraphNode) less(compareTo any) GraphNode {
	return GraphNode{}
}

func (a GraphNode) greater(compareTo any) GraphNode {
	return GraphNode{}
}

func (a GraphNode) equal(compareTo any) GraphNode {
	//switch compareTo.(type) {
	//case GraphNode:
	//	C.fgreater_g(a.ref, compareTo.ref)
	//}
	return GraphNode{}
}

func (a GraphNode) matmul(b GraphNode) GraphNode {
	C.fmatmul(a.ref, b.ref)
	return GraphNode{ref: nil}
}

func (a GraphNode) flatten() GraphNode {
	C.fflatten(a.ref)
	return GraphNode{ref: nil}
}

func (a GraphNode) flattenDim(dim int) GraphNode {
	C.fflatten_dimension(a.ref, C.int(dim))
	return GraphNode{ref: nil}
}

// FIXME: how to do this? can't use generics on receiver functions
func (a GraphNode) convert() GraphNode {
	newType := uint32(C.F_INT32)
	C.fconvert(a.ref, newType)
	return GraphNode{ref: nil}
}

func (a GraphNode) reshape(shape Shape) GraphNode {
	C.freshape(a.ref, nil, C.int(len(shape)))
	return GraphNode{ref: nil}
}

func (a GraphNode) min(b any) GraphNode {
	// FIXME: any has to be float, double, int, long or GraphNode type
	//switch any(b).(type) {
	//case GraphNode:
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
	return GraphNode{ref: nil}
}

func (a GraphNode) max(b any) GraphNode {
	// FIXME: any has to be float, double, int, long or GraphNode type
	//switch b.(type) {
	//case GraphNode:
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
	return GraphNode{ref: nil}
}

func (a GraphNode) reduceSum(dim int) GraphNode {
	C.freduce_sum(a.ref, C.int(dim))
	return GraphNode{ref: nil}
}

func (a GraphNode) reduceMul(dim int) GraphNode {
	C.freduce_mul(a.ref, C.int(dim))
	return GraphNode{ref: nil}
}

func (a GraphNode) reduceMin(dim int) GraphNode {
	C.freduce_min(a.ref, C.int(dim))
	return GraphNode{ref: nil}
}

func (a GraphNode) reduceMax(dim int) GraphNode {
	C.freduce_max(a.ref, C.int(dim))
	return GraphNode{ref: nil}
}

func (a GraphNode) slice(start Axes, end Axes) GraphNode {
	C.fslice(a.ref, nil, nil)
	return GraphNode{ref: nil}
}

func (a GraphNode) sliceWithStride(start Axes, end Axes, stride Stride) GraphNode {
	C.fslice_step(a.ref, nil, nil, nil)
	return GraphNode{ref: nil}
}

func (a GraphNode) extend(shape Shape, insertAt Axes) GraphNode {
	C.fextend(a.ref, nil, nil)
	return GraphNode{ref: nil}
}

func (a GraphNode) extendWithStride(shape Shape, insertAt Axes, stride Stride) GraphNode {
	C.fextend_step(a.ref, nil, nil, nil)
	return GraphNode{ref: nil}
}

func (a GraphNode) concat(b GraphNode, axis uint) GraphNode {
	C.fconcat(a.ref, b.ref, C.uint(axis))
	return GraphNode{ref: nil}
}

func (a GraphNode) expand(axis uint, size uint) GraphNode {
	C.fexpand(a.ref, C.uint(axis), C.uint(size))
	return GraphNode{ref: nil}
}

func (a GraphNode) abs() GraphNode {
	C.fabs_g(a.ref)
	return GraphNode{ref: nil}
}

func (a GraphNode) repeat(repetitions Axes) GraphNode {
	C.frepeat(a.ref, nil)
	return GraphNode{ref: nil}
}

func (a GraphNode) transpose(axes Axes) GraphNode {
	C.ftranspose(a.ref, nil)
	return GraphNode{ref: nil}
}

func (a GraphNode) convolve(kernel GraphNode, stride Stride) GraphNode {
	C.fconvolve(a.ref, kernel.ref, nil)
	return GraphNode{ref: nil}
}

func (a GraphNode) slide(kernel GraphNode, stride Stride) GraphNode {
	C.fslide(a.ref, kernel.ref, nil)
	return GraphNode{ref: nil}
}

func (a GraphNode) index(indices GraphNode) GraphNode {
	C.findex(a.ref, indices.ref)
	return GraphNode{ref: nil}
}

func (a GraphNode) indexSet(b GraphNode, indices GraphNode) GraphNode {
	C.findex_set(a.ref, b.ref, indices.ref)
	return GraphNode{ref: nil}
}

func (a GraphNode) slidingWindow(size Stride, stride Stride) GraphNode {
	C.fsliding_window(a.ref, nil, nil)
	return GraphNode{ref: nil}
}

func (a GraphNode) permute(axis uint) GraphNode {
	C.fpermutate(a.ref, C.uint(axis))
	return GraphNode{ref: nil}
}
