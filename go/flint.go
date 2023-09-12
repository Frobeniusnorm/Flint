// Package flint contains all types and function declarations similar to [../flint.h]
package flint

/*
#cgo LDFLAGS: -lflint -lOpenCL -lstdc++
#include "../flint.h"
*/
import "C"
import (
	"fmt"
	"go/types"
	"unsafe"
)

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

type operationType int

const (
	STORE operationType = iota
	GEN_RANDOM
	GEN_CONSTANT
	GEN_ARANGE
	ADD
	SUB
	MUL
	DIV
	POW
	NEG
	LOG
	SIGN
	EVEN
	LOG2
	LOG10
	SIN
	COS
	TAN
	ASIN
	ACOS
	ATAN
	SQRT
	EXP
	FLATTEN
	MATMUL
	CONVERSION
	RESHAPE
	MIN
	MAX
	REDUCE_SUM
	REDUCE_MUL
	REDUCE_MIN
	REDUCE_MAX
	SLICE
	ABS
	REPEAT
	TRANSPOSE
	EXTEND
	CONCAT
	LESS
	EQUAL
	GREATER
	CONVOLVE
	SLIDE
	GRADIENT_CONVOLVE
	INDEX
	SET_INDEX
	SLIDING_WINDOW
	NUM_OPERATION_TYPES
)

type Shape []uint64

type Operation[T tensorDataType] struct {
	dimension int
	shape     Shape
	opType    operationType
	fpType    T
	extraData []byte
}

type GraphNode[T tensorDataType] struct {
	//predecessors []GraphNode[T]
	//operation    Operation[T]
	// TODO: result data
	// TODO: gradient data
}

// TODO: need to keep a reference to C memory
type tensorDataType interface {
	~int32 | ~int64 | ~float32 | ~float64 | ~int
}

type TensorSlice types.Slice

func CreateGraph[T tensorDataType](data []T, shape Shape) GraphNode[T] {
	//dataPointer := unsafe.Pointer(&data)
	//shapePointer := unsafe.Pointer(&shape)

	var newShape = []C.ulong{C.ulong(2), C.ulong(2)}
	newShapePtr := (*C.ulong)(unsafe.Pointer(&newShape))

	var newData = []C.float{C.float(1.0), C.float(2.0), C.float(3.0), C.float(4.0)}
	newDataPtr := unsafe.Pointer(&newData)

	C.fCreateGraph(newDataPtr, C.int(len(data)), C.F_FLOAT32, newShapePtr, C.int(len(shape)))

	return GraphNode[T]{}
}

func TestGraph() {
	var newShape = []C.ulong{C.ulong(2), C.ulong(2)}
	newShapePtr := (*C.ulong)(unsafe.Pointer(&newShape[0]))

	var newData = []C.float{C.float(1.0), C.float(2.0), C.float(3.0), C.float(4.0)}
	newDataPtr := unsafe.Pointer(&newData[0]) // NOTE: remember to take a pointer to the first element

	structPtr := C.fCreateGraph(newDataPtr, C.int(4), C.F_FLOAT32, newShapePtr, C.int(2))
	fmt.Println("num pre:", structPtr)
}

func cGraphNodeToGo[T tensorDataType](data *C.FGraphNode) GraphNode[T] {
	return GraphNode[T]{}
}

//
//func goShapeToC(shape Shape) *C.ulong {
//	return uintptr(5)
//}
//
//func CreateConstant(value any, shape Shape) {
//	var newShape = []C.ulong{C.ulong(2), C.ulong(2)}
//	newShapePtr := (*C.ulong)(unsafe.Pointer(&newShape[0]))
//
//	var node C.FGraphNode
//	// FIXME: what about unsigned types?
//	switch value.(type) {
//	case int, int32:
//		node = C.fconstant_i(C.int(value), newShapePtr, 2)
//	case int64:
//		node = C.fconstant_l(C.long(value), newShapePtr, 2)
//	case float32:
//		node = C.fconstant_f(C.float(value), newShapePtr, 2)
//	case float64:
//		node = C.fconstant_d(C.double(value), newShapePtr, 2)
//	default:
//		panic("invalid type")
//	}
//
//}
//
//func CreateRandom(shape Shape) GraphNode[float64] {
//	node := C.frandom(goShapeToC(shape), C.int(len(shape)))
//	return cGraphNodeToGo[float64](node)
//}
//
//func CreateArrange(shape Shape) GraphNode[int64] {
//	node := C.farange(goShapeToC(shape), C.int(len(shape)), C.int(1))
//	return cGraphNodeToGo[int64](node)
//}

// TODO: the free method for memory management
// - free Graph
// - sync memory

//func ExecuteGraph[T tensorDataType](node GraphNode[T]) GraphNode[T] {
//}

type Node struct {
	ref   *C.FGraphNode // FIXME: or unsafe pointer?
	shape Shape
}

func (a *Node) div() {

}

type Stride []int
type Axes []int // needs to have one entry for each dimension of tensor

// TODO: slicing

func (a *Node) extend(shape Shape, insertAt Axes) Node {
	C.fextend(a.ref, nil, nil)
	return Node{ref: nil}
}

func (a *Node) extendWithStride(shape Shape, insertAt Axes, stride Stride) Node {
	C.fextend_step(a.ref, nil, nil, nil)
	return Node{ref: nil}
}

func (a *Node) concat(b *Node, axis int) Node {
	C.fconcat(a.ref, b.ref, C.int(axis))
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
