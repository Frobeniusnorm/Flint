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

func Logging(level loggingLevel, message string) {
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

func CreateGraph[T Numeric](data []T, shape Shape) GraphNode {
	datatype := F_FLOAT32
	shapePtr := (*C.size_t)(unsafe.Pointer(&shape[0]))

	dataPtr := unsafe.Pointer(&data[0])
	flintNode := C.fCreateGraph(dataPtr, C.int(len(data)), datatype, shapePtr, C.int(len(shape)))
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

func CreateGraphRandom(shape Shape) GraphNode {
	shapePtr := (*C.size_t)(unsafe.Pointer(&shape[0]))
	flintNode := C.frandom(shapePtr, C.int(len(shape)))
	return GraphNode{ref: flintNode}
}

func CreateGraphArrange(shape Shape) GraphNode {
	shapePtr := (*C.size_t)(unsafe.Pointer(&shape[0]))
	flintNode := C.farange(shapePtr, C.int(len(shape)))
	return GraphNode{ref: flintNode}
}

func (a GraphNode) Result() Result {
	flintNode := C.fCalculateResult(a.ref)
	return Result{
		//shape: cToShape(flintNode.operation.shape),
		data: flintNode.result_data.data,
	}
}

type GradientContext struct{}

func (_ GradientContext) Start() {
	C.fStartGradientContext()
}

func (_ GradientContext) Stop() {
	C.fStopGradientContext()
}

func (_ GradientContext) Active() bool {
	res := C.fIsGradientContext()
	fmt.Println("is grad ctx:", res)
	return false
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

func Pow[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch c := any(b).(type) {
	case GraphNode:
		flintNode = C.fpow_g(a.ref, c.ref)
	case int32, int:
		flintNode = C.fpow_ci(a.ref, C.int(c))
	case int64:
		flintNode = C.fpow_cl(a.ref, C.long(c))
	case float32:
		flintNode = C.fpow_cf(a.ref, C.float(c))
	case float64:
		flintNode = C.fpow_cd(a.ref, C.double(c))
	default:
		panic("invalid type")
	}
	return GraphNode{ref: flintNode}
}

func Mul[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch c := any(b).(type) {
	case GraphNode:
		flintNode = C.fmul_g(a.ref, c.ref)
	case int32, int:
		flintNode = C.fmul_ci(a.ref, C.int(c))
	case int64:
		flintNode = C.fmul_cl(a.ref, C.long(c))
	case float32:
		flintNode = C.fmul_cf(a.ref, C.float(c))
	case float64:
		flintNode = C.fmul_cd(a.ref, C.double(c))
	default:
		panic("invalid type")
	}
	return GraphNode{ref: flintNode}
}

// Div Divides a by b.
// At least one of the parameters HAS to be a GraphNode
// Division with Tensors is carried out element-wise
func Div[T Numeric | GraphNode](a T, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch x := any(a).(type) {
	case GraphNode:
		switch y := any(b).(type) {
		case GraphNode:
			flintNode = C.fdiv_g(x.ref, y.ref)
		case int32, int:
			flintNode = C.fdiv_ci(x.ref, C.int(y))
		case int64:
			flintNode = C.fdiv_cl(x.ref, C.long(y))
		case float32:
			flintNode = C.fdiv_cf(x.ref, C.float(y))
		case float64:
			flintNode = C.fdiv_cd(x.ref, C.double(y))
		default:
			panic("invalid type")
		}
	case int32, int:
		if y, isNode := any(b).(GraphNode); isNode == true {
			flintNode = C.fdiv_ici(C.int(x), y.ref)
		} else {
			panic("invalid type")
		}
	case int64:
		if y, isNode := any(b).(GraphNode); isNode == true {
			flintNode = C.fdiv_icl(C.int(x), y.ref)
		} else {
			panic("invalid type")
		}
	case float32:
		if y, isNode := any(b).(GraphNode); isNode == true {
			flintNode = C.fdiv_icf(C.int(x), y.ref)
		} else {
			panic("invalid type")
		}
	case float64:
		if y, isNode := any(b).(GraphNode); isNode == true {
			flintNode = C.fdiv_icl(C.int(x), y.ref)
		} else {
			panic("invalid type")
		}
	default:
		panic("invalid type")
	}
	return GraphNode{ref: flintNode}
}

// Sub Subtracts a by b.
// At least one of the parameters HAS to be a GraphNode
// Subtraction with Tensors is carried out element-wise
func Sub[T Numeric | GraphNode](a T, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch x := any(a).(type) {
	case GraphNode:
		switch y := any(b).(type) {
		case GraphNode:
			flintNode = C.fsub_g(x.ref, y.ref)
		case int32, int:
			flintNode = C.fsub_ci(x.ref, C.int(y))
		case int64:
			flintNode = C.fsub_cl(x.ref, C.long(y))
		case float32:
			flintNode = C.fsub_cf(x.ref, C.float(y))
		case float64:
			flintNode = C.fsub_cd(x.ref, C.double(y))
		default:
			panic("invalid type")
		}
	case int32, int:
		if y, isNode := any(b).(GraphNode); isNode == true {
			flintNode = C.fsub_ici(C.int(x), y.ref)
		} else {
			panic("invalid type")
		}
	case int64:
		if y, isNode := any(b).(GraphNode); isNode == true {
			flintNode = C.fsub_icl(C.int(x), y.ref)
		} else {
			panic("invalid type")
		}
	case float32:
		if y, isNode := any(b).(GraphNode); isNode == true {
			flintNode = C.fsub_icf(C.int(x), y.ref)
		} else {
			panic("invalid type")
		}
	case float64:
		if y, isNode := any(b).(GraphNode); isNode == true {
			flintNode = C.fsub_icl(C.int(x), y.ref)
		} else {
			panic("invalid type")
		}
	default:
		panic("invalid type")
	}
	return GraphNode{ref: flintNode}
}

func Log(a GraphNode) GraphNode {
	flintNode := C.flog(a.ref)
	return GraphNode{ref: flintNode}
}

func Log2(a GraphNode) GraphNode {
	flintNode := C.flog2(a.ref)
	return GraphNode{ref: flintNode}
}

func Log10(a GraphNode) GraphNode {
	flintNode := C.flog10(a.ref)
	return GraphNode{ref: flintNode}
}

func Sin(a GraphNode) GraphNode {
	flintNode := C.fsin(a.ref)
	return GraphNode{ref: flintNode}
}

func Sqrt(a GraphNode) GraphNode {
	flintNode := C.fsqrt_g(a.ref)
	return GraphNode{ref: flintNode}
}

func Exp(a GraphNode) GraphNode {
	flintNode := C.fexp(a.ref)
	return GraphNode{ref: flintNode}
}

func Cos(a GraphNode) GraphNode {
	flintNode := C.fcos(a.ref)
	return GraphNode{ref: flintNode}
}

func Tan(a GraphNode) GraphNode {
	flintNode := C.ftan(a.ref)
	return GraphNode{ref: flintNode}
}

func Asin(a GraphNode) GraphNode {
	flintNode := C.fasin(a.ref)
	return GraphNode{ref: flintNode}
}

func Acos(a GraphNode) GraphNode {
	flintNode := C.facos(a.ref)
	return GraphNode{ref: flintNode}
}

func Atan(a GraphNode) GraphNode {
	flintNode := C.fatan(a.ref)
	return GraphNode{ref: flintNode}
}

func Neg(a GraphNode) GraphNode {
	flintNode := C.fneg(a.ref)
	return GraphNode{ref: flintNode}
}

func Sign(a GraphNode) GraphNode {
	flintNode := C.fsign(a.ref)
	return GraphNode{ref: flintNode}
}

func Even(a GraphNode) GraphNode {
	flintNode := C.feven(a.ref)
	return GraphNode{ref: flintNode}
}

func Equal[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch c := any(b).(type) {
	case GraphNode:
		flintNode = C.fequal_g(a.ref, c.ref)
	case int32, int:
		flintNode = C.fequal_ci(a.ref, C.int(c))
	case int64:
		flintNode = C.fequal_cl(a.ref, C.long(c))
	case float32:
		flintNode = C.fequal_cf(a.ref, C.float(c))
	case float64:
		flintNode = C.fequal_cd(a.ref, C.double(c))
	default:
		panic("invalid type")
	}
	return GraphNode{ref: flintNode}
}

func Greater[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch c := any(b).(type) {
	case GraphNode:
		flintNode = C.fgreater_g(a.ref, c.ref)
	case int32, int:
		flintNode = C.fgreater_ci(a.ref, C.int(c))
	case int64:
		flintNode = C.fgreater_cl(a.ref, C.long(c))
	case float32:
		flintNode = C.fgreater_cf(a.ref, C.float(c))
	case float64:
		flintNode = C.fgreater_cd(a.ref, C.double(c))
	default:
		panic("invalid type")
	}
	return GraphNode{ref: flintNode}
}

func Less[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch c := any(b).(type) {
	case GraphNode:
		flintNode = C.fless_g(a.ref, c.ref)
	case int32, int:
		flintNode = C.fless_ci(a.ref, C.int(c))
	case int64:
		flintNode = C.fless_cl(a.ref, C.long(c))
	case float32:
		flintNode = C.fless_cf(a.ref, C.float(c))
	case float64:
		flintNode = C.fless_cd(a.ref, C.double(c))
	default:
		panic("invalid type")
	}
	return GraphNode{ref: flintNode}
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
	newTypeXXXXX := uint32(C.F_INT32)
	flintNode := C.fconvert(a.ref, newTypeXXXXX)
	return GraphNode{ref: flintNode}
}

func Reshape(a GraphNode, shape Shape) GraphNode {
	flintNode := C.freshape(a.ref, nil, C.int(len(shape)))
	return GraphNode{ref: flintNode}
}

func Min[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch c := any(b).(type) {
	case GraphNode:
		flintNode = C.fmin_g(a.ref, c.ref)
	case int32, int:
		flintNode = C.fmin_ci(a.ref, C.int(c))
	case int64:
		flintNode = C.fmin_cl(a.ref, C.long(c))
	case float32:
		flintNode = C.fmin_cf(a.ref, C.float(c))
	case float64:
		flintNode = C.fmin_cd(a.ref, C.double(c))
	default:
		panic("invalid type")
	}
	return GraphNode{ref: flintNode}
}

func Max[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch c := any(b).(type) {
	case GraphNode:
		flintNode = C.fmax_g(a.ref, c.ref)
	case int32, int:
		flintNode = C.fmax_ci(a.ref, C.int(c))
	case int64:
		flintNode = C.fmax_cl(a.ref, C.long(c))
	case float32:
		flintNode = C.fmax_cf(a.ref, C.float(c))
	case float64:
		flintNode = C.fmax_cd(a.ref, C.double(c))
	default:
		panic("invalid type")
	}
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

func SlidingWindow(a GraphNode, size Shape, stride Stride) GraphNode {
	flintNode := C.fsliding_window(a.ref, nil, nil)
	return GraphNode{ref: flintNode}
}

func Permute(a GraphNode, axis uint) GraphNode {
	flintNode := C.fpermutate(a.ref, C.uint(axis))
	return GraphNode{ref: flintNode}
}
