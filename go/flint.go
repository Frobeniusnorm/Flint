// Package flint contains all types and function declarations similar to [../flint.h]
package flint

/*
* important rules for writing robust CGo code:
* - only pass C types to C
* - dont pass data from Go's stack memory to C. include stdlib and use C.malloc!
*/

/*
#cgo LDFLAGS: -lflint -lOpenCL -lstdc++ -lm
#include <flint/flint.h>
#include <stdlib.h>  // needed for C.free!
*/
import "C"
import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"
)

///////////////
// Types and Structs
//////////////

type Tensor[T Numeric] struct {
	data  []T
	shape Shape
}
type FloatTensor Tensor[float32]
type IntTensor Tensor[int32]
type DoubleTensor Tensor[float64]
type LongTensor Tensor[int64]

// GraphNode points to a FGraphNode in C Heap memory
type GraphNode unsafe.Pointer

func graphRef(a GraphNode) *C.FGraphNode {
	return (*C.FGraphNode)(a)
}

type Numeric interface {
	~int32 | ~int64 | ~float32 | ~float64
}

type Stride []int // needs to have one entry for each dimension of tensor
type Axes []uint  // needs to have one entry for each dimension of tensor
type Shape []uint // needs to have one entry for each dimension of tensor

type tensorDataType uint32

const (
	F_INT32 tensorDataType = iota
	F_INT64
	F_FLOAT32
	F_FLOAT64
)

type completeNumbers interface {
	Numeric | ~uint | ~int | ~int8 | ~int64 | ~uint64 | ~uint16 | ~uint8
}

// cNumbers represents the usable C types
// size_t is equivalent to ulong
type cNumbers interface {
	C.int | C.size_t | C.long | C.uint | C.float | C.double
}

func convertArray[In completeNumbers, Out completeNumbers | cNumbers](arr []In) []Out {
	result := make([]Out, len(arr))
	for idx, val := range arr {
		result[idx] = Out(val)
	}
	return result
}

//////////////
// SETUP
/////////////

type Backend int

const (
	BACKEND_ONLY_CPU Backend = iota + 1
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

func FreeGraph(a GraphNode) {
	C.fFreeGraph(graphRef(a))
}

func CopyGraph(a GraphNode) GraphNode {
	flintNode := C.fCopyGraph(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
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

// ////////////
// Images
// ///////////
type imageFormat int

const (
	PNG imageFormat = iota
	JPEG
	BMP
)

func LoadImage(path string) GraphNode {
	unsafePath := C.CString(path)
	defer C.free(unsafe.Pointer(unsafePath))
	flintNode, err := C.fload_image(unsafePath)
	if err != nil {
		panic(err) // TODO: error handling everywhere
	}
	return GraphNode(unsafe.Pointer(flintNode))
}

func StoreImage(node GraphNode, path string, format imageFormat) {
	// FIXME: jpeg writing is broken
	unsafePath := C.CString(path)
	defer C.free(unsafe.Pointer(unsafePath))
	C.fstore_image(graphRef(node), unsafePath, C.enum_FImageFormat(format))
}

// /////////////
// Utility functions
// ////////////

func GetShape(node GraphNode) Shape {
	flintNode := graphRef(node)
	shapePtr := unsafe.Pointer(flintNode.operation.shape)
	shapeSize := int(flintNode.operation.dimensions)
	return fromCToArray[uint](shapePtr, shapeSize, F_INT64)
}

// use for type debugging
func describe(i any) {
	fmt.Printf("describe (value, underlying type): (%v, %T)\n", i, i)
}

//func arrayFromC[T Numeric | uint64 | uint32 | int | uint](length int, dataPtr unsafe.Pointer, dataType tensorDataType) []T {
//	var sizeOf int
//	switch dataType {
//	case F_INT32:
//		sizeOf = int(C.sizeof_int)
//	case F_INT64:
//		sizeOf = int(C.sizeof_long)
//	case F_FLOAT32:
//		sizeOf = int(C.sizeof_float)
//	case F_FLOAT64:
//		sizeOf = int(C.sizeof_double)
//	default:
//		panic("invalid type")
//	}
//
//	var result = make([]T, length)
//	//voidPtr := (*C.void)(dataPtr)
//	for i := 0; i < length; i++ {
//		//x := *(voidPtr + C.int(i)) // add sizeof?
//		//x := unsafe.Pointer(dataPtr + C.int(i))
//		x := dataPtr + i*sizeOf
//		result[i] = T(x)
//	}
//	return result
//}

func fromCToArray[T completeNumbers](dataPtr unsafe.Pointer, length int, dataType tensorDataType) []T {
	var result = make([]T, length)

	switch dataType {
	case F_INT32:
		var data []byte = C.GoBytes(dataPtr, C.int(length*C.sizeof_int))
		elementSize := len(data) / length
		for i := 0; i < length; i++ {
			relevantData := data[i*elementSize : i*elementSize+elementSize]
			x := binary.LittleEndian.Uint32(relevantData)
			result[i] = T(x)
		}
	case F_INT64:
		var data []byte = C.GoBytes(dataPtr, C.int(length*C.sizeof_long))
		elementSize := len(data) / length
		for i := 0; i < length; i++ {
			relevantData := data[i*elementSize : i*elementSize+elementSize]
			x := binary.LittleEndian.Uint64(relevantData)
			result[i] = T(x)
		}

	case F_FLOAT32:
		var data []byte = C.GoBytes(dataPtr, C.int(length*C.sizeof_float))
		elementSize := len(data) / length
		for i := 0; i < length; i++ {
			relevantData := data[i*elementSize : i*elementSize+elementSize]
			bits := binary.LittleEndian.Uint32(relevantData)
			x := math.Float32frombits(bits)
			result[i] = T(x)
		}

	case F_FLOAT64:
		var data []byte = C.GoBytes(dataPtr, C.int(length*C.sizeof_double))
		elementSize := len(data) / length
		for i := 0; i < length; i++ {
			relevantData := data[i*elementSize : i*elementSize+elementSize]
			bits := binary.LittleEndian.Uint64(relevantData)
			x := math.Float64frombits(bits)
			result[i] = T(x)
		}
	default:
		panic("invalid type")
	}

	return result
}

// /////////////
// Creating and Executing Tensors
// ////////////

func CreateGraph[T Numeric](data []T, shape Shape) GraphNode {
	// FIXME: support the other data types
	datatype := F_FLOAT32
	newShape := convertArray[uint, C.size_t](shape)
	newData := convertArray[T, C.float](data)

	flintNode := C.fCreateGraph(unsafe.Pointer(&(newData[0])), C.int(len(data)), uint32(datatype), &(newShape[0]), C.int(len(shape)))
	return GraphNode(unsafe.Pointer(flintNode))
}

func CreateGraphConstant[T Numeric](value T, shape Shape) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)

	var flintNode *C.FGraphNode
	dimensions := C.int(len(shape))
	switch v := any(value).(type) {
	case int32:
		flintNode = C.fconstant_i(C.int(v), &(newShape[0]), dimensions)
	case int64:
		flintNode = C.fconstant_l(C.long(v), &(newShape[0]), dimensions)
	case float32:
		flintNode = C.fconstant_f(C.float(v), &(newShape[0]), dimensions)
	case float64:
		flintNode = C.fconstant_d(C.double(v), &(newShape[0]), dimensions)
	default:
		panic("invalid data type")
	}
	return GraphNode(flintNode)
}

func CreateGraphRandom(shape Shape) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)

	flintNode := C.frandom(&(newShape[0]), C.int(len(shape)))
	return GraphNode(unsafe.Pointer(flintNode))
}

func CreateGraphArrange(shape Shape, axis int) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)

	flintNode := C.farange(&(newShape[0]), C.int(len(shape)), C.int(axis))
	return GraphNode(unsafe.Pointer(flintNode))
}

// CalculateResult essentially combines ExecuteGraph and SyncMemory
func CalculateResult[T Numeric](a GraphNode) Tensor[T] {
	flintNode := C.fCalculateResult(graphRef(a))

	dataSize := int(flintNode.result_data.num_entries)
	dataPtr := unsafe.Pointer(flintNode.result_data.data)
	shapePtr := unsafe.Pointer(flintNode.operation.shape)
	shapeSize := int(flintNode.operation.dimensions)

	dataType := tensorDataType(flintNode.operation.data_type)

	var result = fromCToArray[T](dataPtr, dataSize, dataType)
	var shape = Shape(fromCToArray[uint](shapePtr, shapeSize, F_INT64))

	return Tensor[T]{
		data:  result,
		shape: shape,
	}
}

func MarkGradientVariable(a GraphNode) {
	C.fMarkGradientVariable(graphRef(a))
}

func UnmarkGradientVariable(a GraphNode) {
	C.fUnmarkGradientVariable(graphRef(a))
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
	return bool(res)
}

func Serialize(a GraphNode) []byte {
	var size C.size_t
	ptr := C.fserialize(graphRef(a), &size)
	defer C.free(unsafe.Pointer(ptr))
	return C.GoBytes(unsafe.Pointer(ptr), C.int(size))
}

func Deserialize(data []byte) GraphNode {
	unsafeData := C.CBytes(data)
	defer C.free(unsafe.Pointer(unsafeData))
	// this cast is necessary as the binay data needs to be passed as char *.
	flintNode := C.fdeserialize((*C.char)(unsafeData))
	return GraphNode(unsafe.Pointer(flintNode))
}

func IncreaseRefCounter(node GraphNode) {
	ref := (*graphRef(node)).reference_counter
	ref += 1
}

func DecreaseRefCounter(node GraphNode) {
	ref := (*graphRef(node)).reference_counter
	ref -= 1
}

func ExecuteGraph(node GraphNode) GraphNode {
	flintNode := C.fExecuteGraph(graphRef(node))
	return GraphNode(unsafe.Pointer(flintNode))
}

// TODO: fsyncmemory

func OptimizeMemory(node GraphNode) GraphNode {
	flintNode := C.fOptimizeMemory(graphRef(node))
	return GraphNode(unsafe.Pointer(flintNode))
}

///////////////
/// Tensor Operations
//////////////

func Add[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode

	switch c := any(b).(type) {
	case int32:
		flintNode = C.fadd_ci(graphRef(a), C.int(c))
	case int64:
		flintNode = C.fadd_cl(graphRef(a), C.long(c))
	case float32:
		flintNode = C.fadd_cf(graphRef(a), C.float(c))
	case float64:
		flintNode = C.fadd_cd(graphRef(a), C.double(c))
	case GraphNode:
		flintNode = C.fadd_g(graphRef(a), graphRef(c))
	default:
		panic("invalid type")
	}

	return GraphNode(unsafe.Pointer(flintNode))
}

func Pow[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch c := any(b).(type) {
	case GraphNode:
		flintNode = C.fpow_g(graphRef(a), graphRef(c))
	case int32:
		flintNode = C.fpow_ci(graphRef(a), C.int(c))
	case int64:
		flintNode = C.fpow_cl(graphRef(a), C.long(c))
	case float32:
		flintNode = C.fpow_cf(graphRef(a), C.float(c))
	case float64:
		flintNode = C.fpow_cd(graphRef(a), C.double(c))
	default:
		panic("invalid type")
	}
	return GraphNode(unsafe.Pointer(flintNode))
}

func Mul[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch c := any(b).(type) {
	case GraphNode:
		flintNode = C.fmul_g(graphRef(a), graphRef(c))
	case int32:
		flintNode = C.fmul_ci(graphRef(a), C.int(c))
	case int64:
		flintNode = C.fmul_cl(graphRef(a), C.long(c))
	case float32:
		flintNode = C.fmul_cf(graphRef(a), C.float(c))
	case float64:
		flintNode = C.fmul_cd(graphRef(a), C.double(c))
	default:
		panic("invalid type")
	}
	return GraphNode(unsafe.Pointer(flintNode))
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
			flintNode = C.fdiv_g(graphRef(x), graphRef(y))
		case int32:
			flintNode = C.fdiv_ci(graphRef(x), C.int(y))
		case int64:
			flintNode = C.fdiv_cl(graphRef(x), C.long(y))
		case float32:
			flintNode = C.fdiv_cf(graphRef(x), C.float(y))
		case float64:
			flintNode = C.fdiv_cd(graphRef(x), C.double(y))
		default:
			panic("invalid type")
		}
	case int32:
		if y, isNode := any(b).(GraphNode); isNode == true {
			flintNode = C.fdiv_ici(C.int(x), graphRef(y))
		} else {
			panic("invalid type")
		}
	case int64:
		if y, isNode := any(b).(GraphNode); isNode == true {
			flintNode = C.fdiv_icl(C.long(x), graphRef(y))
		} else {
			panic("invalid type")
		}
	case float32:
		if y, isNode := any(b).(GraphNode); isNode == true {
			flintNode = C.fdiv_icf(C.float(x), graphRef(y))
		} else {
			panic("invalid type")
		}
	case float64:
		if y, isNode := any(b).(GraphNode); isNode == true {
			flintNode = C.fdiv_icd(C.double(x), graphRef(y))
		} else {
			panic("invalid type")
		}
	default:
		panic("invalid type")
	}
	return GraphNode(flintNode)
}

// Sub Subtracts graphRef(a) by b.
// At least one of the parameters HAS to be graphRef(a) GraphNode
// Subtraction with Tensors is carried out element-wise
func Sub[T Numeric | GraphNode](a T, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch x := any(a).(type) {
	case GraphNode:
		switch y := any(b).(type) {
		case GraphNode:
			flintNode = C.fsub_g(graphRef(x), graphRef(y))
		case int32:
			flintNode = C.fsub_ci(graphRef(x), C.int(y))
		case int64:
			flintNode = C.fsub_cl(graphRef(x), C.long(y))
		case float32:
			flintNode = C.fsub_cf(graphRef(x), C.float(y))
		case float64:
			flintNode = C.fsub_cd(graphRef(x), C.double(y))
		default:
			panic("invalid type")
		}
	case int32:
		if y, isNode := any(b).(GraphNode); isNode == true {
			flintNode = C.fsub_ici(C.int(x), graphRef(y))
		} else {
			panic("invalid type")
		}
	case int64:
		if y, isNode := any(b).(GraphNode); isNode == true {
			flintNode = C.fsub_icl(C.long(x), graphRef(y))
		} else {
			panic("invalid type")
		}
	case float32:
		if y, isNode := any(b).(GraphNode); isNode == true {
			flintNode = C.fsub_icf(C.float(x), graphRef(y))
		} else {
			panic("invalid type")
		}
	case float64:
		if y, isNode := any(b).(GraphNode); isNode == true {
			flintNode = C.fsub_icd(C.double(x), graphRef(y))
		} else {
			panic("invalid type")
		}
	default:
		panic("invalid type")
	}
	return GraphNode(unsafe.Pointer(flintNode))
}

func Log(a GraphNode) GraphNode {
	flintNode := C.flog(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Log2(a GraphNode) GraphNode {
	flintNode := C.flog2(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Log10(a GraphNode) GraphNode {
	flintNode := C.flog10(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Sin(a GraphNode) GraphNode {
	flintNode := C.fsin(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Sqrt(a GraphNode) GraphNode {
	flintNode := C.fsqrt_g(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Exp(a GraphNode) GraphNode {
	flintNode := C.fexp(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Cos(a GraphNode) GraphNode {
	flintNode := C.fcos(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Tan(a GraphNode) GraphNode {
	flintNode := C.ftan(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Asin(a GraphNode) GraphNode {
	flintNode := C.fasin(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Acos(a GraphNode) GraphNode {
	flintNode := C.facos(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Atan(a GraphNode) GraphNode {
	flintNode := C.fatan(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Neg(a GraphNode) GraphNode {
	flintNode := C.fneg(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Sign(a GraphNode) GraphNode {
	flintNode := C.fsign(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Even(a GraphNode) GraphNode {
	flintNode := C.feven(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Equal[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch c := any(b).(type) {
	case GraphNode:
		flintNode = C.fequal_g(graphRef(a), graphRef(c))
	case int32:
		flintNode = C.fequal_ci(graphRef(a), C.int(c))
	case int64:
		flintNode = C.fequal_cl(graphRef(a), C.long(c))
	case float32:
		flintNode = C.fequal_cf(graphRef(a), C.float(c))
	case float64:
		flintNode = C.fequal_cd(graphRef(a), C.double(c))
	default:
		panic("invalid type")
	}
	return GraphNode(unsafe.Pointer(flintNode))
}

func Greater[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch c := any(b).(type) {
	case GraphNode:
		flintNode = C.fgreater_g(graphRef(a), graphRef(c))
	case int32:
		flintNode = C.fgreater_ci(graphRef(a), C.int(c))
	case int64:
		flintNode = C.fgreater_cl(graphRef(a), C.long(c))
	case float32:
		flintNode = C.fgreater_cf(graphRef(a), C.float(c))
	case float64:
		flintNode = C.fgreater_cd(graphRef(a), C.double(c))
	default:
		panic("invalid type")
	}
	return GraphNode(flintNode)
}

func Less[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch c := any(b).(type) {
	case GraphNode:
		flintNode = C.fless_g(graphRef(a), graphRef(c))
	case int32:
		flintNode = C.fless_ci(graphRef(a), C.int(c))
	case int64:
		flintNode = C.fless_cl(graphRef(a), C.long(c))
	case float32:
		flintNode = C.fless_cf(graphRef(a), C.float(c))
	case float64:
		flintNode = C.fless_cd(graphRef(a), C.double(c))
	default:
		panic("invalid type")
	}
	return GraphNode(unsafe.Pointer(flintNode))
}

func Matmul(a GraphNode, b GraphNode) GraphNode {
	flintNode := C.fmatmul(graphRef(a), graphRef(b))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Flatten(a GraphNode) GraphNode {
	flintNode := C.fflatten(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
}

func FlattenDim(a GraphNode, dim int) GraphNode {
	flintNode := C.fflatten_dimension(graphRef(a), C.int(dim))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Convert(a GraphNode, newType tensorDataType) GraphNode {
	flintNode := C.fconvert(graphRef(a), C.enum_FType(newType))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Reshape(a GraphNode, shape Shape) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)

	flintNode := C.freshape(graphRef(a), &(newShape[0]), C.int(len(shape)))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Min[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch c := any(b).(type) {
	case GraphNode:
		flintNode = C.fmin_g(graphRef(a), graphRef(c))
	case int32:
		flintNode = C.fmin_ci(graphRef(a), C.int(c))
	case int64:
		flintNode = C.fmin_cl(graphRef(a), C.long(c))
	case float32:
		flintNode = C.fmin_cf(graphRef(a), C.float(c))
	case float64:
		flintNode = C.fmin_cd(graphRef(a), C.double(c))
	default:
		panic("invalid type")
	}
	return GraphNode(unsafe.Pointer(flintNode))
}

func Max[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode = nil // FIXME: can i replace this type var?
	switch c := any(b).(type) {
	case GraphNode:
		flintNode = C.fmax_g(graphRef(a), graphRef(c))
	case int32:
		flintNode = C.fmax_ci(graphRef(a), C.int(c))
	case int64:
		flintNode = C.fmax_cl(graphRef(a), C.long(c))
	case float32:
		flintNode = C.fmax_cf(graphRef(a), C.float(c))
	case float64:
		flintNode = C.fmax_cd(graphRef(a), C.double(c))
	default:
		panic("invalid type")
	}
	return GraphNode(flintNode)
}

func ReduceSum(a GraphNode, dim int) GraphNode {
	flintNode := C.freduce_sum(graphRef(a), C.int(dim))
	return GraphNode(unsafe.Pointer(flintNode))
}

func ReduceMul(a GraphNode, dim int) GraphNode {
	flintNode := C.freduce_mul(graphRef(a), C.int(dim))
	return GraphNode(unsafe.Pointer(flintNode))
}

func ReduceMin(a GraphNode, dim int) GraphNode {
	flintNode := C.freduce_min(graphRef(a), C.int(dim))
	return GraphNode(unsafe.Pointer(flintNode))
}

func ReduceMax(a GraphNode, dim int) GraphNode {
	flintNode := C.freduce_max(graphRef(a), C.int(dim))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Slice(a GraphNode, start Axes, end Axes) GraphNode {
	newStart := convertArray[uint, C.long](start)
	newEnd := convertArray[uint, C.long](end)

	flintNode := C.fslice(graphRef(a), &(newStart[0]), &(newEnd[0]))
	return GraphNode(unsafe.Pointer(flintNode))
}

func SliceWithStride(a GraphNode, start Axes, end Axes, stride Stride) GraphNode {
	newStart := convertArray[uint, C.long](start)
	newEnd := convertArray[uint, C.long](end)
	newStride := convertArray[int, C.long](stride)

	flintNode := C.fslice_step(graphRef(a), &(newStart[0]), &(newEnd[0]), &(newStride[0]))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Extend(a GraphNode, shape Shape, insertAt Axes) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)
	newInsertAt := convertArray[uint, C.size_t](insertAt)

	flintNode := C.fextend(graphRef(a), &(newShape[0]), &(newInsertAt[0]))
	return GraphNode(unsafe.Pointer(flintNode))
}

func ExtendWithStride(a GraphNode, shape Shape, insertAt Axes, stride Stride) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)
	newInsertAt := convertArray[uint, C.size_t](insertAt)
	newStride := convertArray[int, C.long](stride)

	flintNode := C.fextend_step(graphRef(a), &(newShape[0]), &(newInsertAt[0]), &(newStride[0]))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Concat(a GraphNode, b GraphNode, axis uint) GraphNode {
	flintNode := C.fconcat(graphRef(a), graphRef(b), C.uint(axis))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Expand(a GraphNode, axis uint, size uint) GraphNode {
	flintNode := C.fexpand(graphRef(a), C.uint(axis), C.uint(size))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Abs(a GraphNode) GraphNode {
	flintNode := C.fabs_g(graphRef(a))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Repeat(a GraphNode, repetitions Axes) GraphNode {
	newRepetitions := convertArray[uint, C.int](repetitions)

	flintNode := C.frepeat(graphRef(a), &(newRepetitions[0]))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Transpose(a GraphNode, axes Axes) GraphNode {
	newAxes := convertArray[uint, C.int](axes)

	flintNode := C.ftranspose(graphRef(a), &(newAxes[0]))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Convolve(a GraphNode, kernel GraphNode, stride Stride) GraphNode {
	newStride := convertArray[int, C.uint](stride)

	flintNode := C.fconvolve(graphRef(a), graphRef(kernel), &(newStride[0]))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Slide(a GraphNode, kernel GraphNode, stride Stride) GraphNode {
	newStride := convertArray[int, C.uint](stride)

	flintNode := C.fslide(graphRef(a), graphRef(kernel), &(newStride[0]))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Index(a GraphNode, indices GraphNode) GraphNode {
	flintNode := C.findex(graphRef(a), graphRef(indices))
	return GraphNode(unsafe.Pointer(flintNode))
}

func IndexSet(a GraphNode, b GraphNode, indices GraphNode) GraphNode {
	flintNode := C.findex_set(graphRef(a), graphRef(b), graphRef(indices))
	return GraphNode(unsafe.Pointer(flintNode))
}

func SlidingWindow(a GraphNode, size Shape, stride Stride) GraphNode {
	newSize := convertArray[uint, C.size_t](size)
	newStride := convertArray[int, C.uint](stride)

	flintNode := C.fsliding_window(graphRef(a), &(newSize[0]), &(newStride[0]))
	return GraphNode(unsafe.Pointer(flintNode))
}

func Permute(a GraphNode, axis uint) GraphNode {
	flintNode := C.fpermutate(graphRef(a), C.uint(axis))
	return GraphNode(unsafe.Pointer(flintNode))
}
