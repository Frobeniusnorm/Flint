// Package flint contains all basic types and functions
// as it is only the low-level wrapper, declarations similar to [../flint.h]
//
// this wrapper is features complete, but should only be used as a basis for higher level software,
// as there is no memory management or similar things.
//
// [../flint.h]
package flint

/*
NOTE: important rules for writing robust CGo code:
- only pass C types to C!
- dont pass data from Go's stack memory to C. include stdlib and use C.malloc!
- only pass unsafe.Pointer and C pointer to cgo!
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

// Tensor is a higher level abstraction of a GraphNode. It includes the data using Go types
// The zero value is only given when the execution did not yield any result.
type Tensor[T Numeric] struct {
	data  []T
	shape Shape
}
type FloatTensor Tensor[float32]
type IntTensor Tensor[int32]
type DoubleTensor Tensor[float64]
type LongTensor Tensor[int64]

// A GraphNode is just a points to a FGraphNode in C Heap memory
type GraphNode unsafe.Pointer

// graphRef turns the unsafe pointer into a C pointer
// As C Types should not be exported, this function HAS to stay local
func graphRef(node GraphNode) *C.FGraphNode {
	return (*C.FGraphNode)(node)
}

// Numeric is a constraint interface representing the supported types of C operations
// TODO: as we no longer cast values between C and GO (we shouldn't!) this can be extended!
type Numeric interface {
	~int32 | ~int64 | ~float32 | ~float64
}

// Stride defines the steps for sliding operations
// needs to have one entry for each dimension of tensor
type Stride []int

// Axes indicate changes in dimensions (i.e transpose)
// needs to have one entry for each dimension of tensor
// and each entry should not be higher than the number of dimensions
type Axes []uint

// Shape represents the size of a tensor.
// needs to have one entry for each dimension of tensor
type Shape []uint

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

func FreeGraph(node GraphNode) {
	C.fFreeGraph(graphRef(node))
}

func CopyGraph(node GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fCopyGraph(graphRef(node))
	return GraphNode(flintNode)
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
	var flintNode *C.FGraphNode = graphRef(node)
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

	var flintNode *C.FGraphNode = C.fCreateGraph(unsafe.Pointer(&(newData[0])), C.int(len(data)), uint32(datatype), &(newShape[0]), C.int(len(shape)))
	return GraphNode(flintNode)
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

	var flintNode *C.FGraphNode = C.frandom(&(newShape[0]), C.int(len(shape)))
	return GraphNode(flintNode)
}

func CreateGraphArrange(shape Shape, axis int) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)

	var flintNode *C.FGraphNode = C.farange(&(newShape[0]), C.int(len(shape)), C.int(axis))
	return GraphNode(flintNode)
}

// CalculateResult essentially combines ExecuteGraph and SyncMemory
func CalculateResult[T Numeric](node GraphNode) Tensor[T] {
	var flintNode *C.FGraphNode = C.fCalculateResult(graphRef(node))

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

func MarkGradientVariable(node GraphNode) {
	C.fMarkGradientVariable(graphRef(node))
}

func UnmarkGradientVariable(node GraphNode) {
	C.fUnmarkGradientVariable(graphRef(node))
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

func Serialize(node GraphNode) []byte {
	var size C.size_t
	ptr := C.fserialize(graphRef(node), &size)
	defer C.free(unsafe.Pointer(ptr))
	return C.GoBytes(unsafe.Pointer(ptr), C.int(size))
}

func Deserialize(data []byte) GraphNode {
	unsafeData := C.CBytes(data)
	defer C.free(unsafe.Pointer(unsafeData))
	// this cast is necessary as the binay data needs to be passed as char *.
	var flintNode *C.FGraphNode = C.fdeserialize((*C.char)(unsafeData))
	return GraphNode(flintNode)
}

func IncreaseRefCounter(node GraphNode) {
	var flintNode *C.FGraphNode = graphRef(node)
	flintNode.reference_counter = C.size_t(flintNode.reference_counter + C.size_t(1))
}

func DecreaseRefCounter(node GraphNode) {
	var flintNode *C.FGraphNode = graphRef(node)
	// make sure it does not underflow!
	if uint64(flintNode.reference_counter) == 0 {
		return
	}
	flintNode.reference_counter = C.size_t(flintNode.reference_counter - C.size_t(1))
}

func ExecuteGraph(node GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fExecuteGraph(graphRef(node))
	return GraphNode(flintNode)
}

// TODO: fsyncmemory

func OptimizeMemory(node GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fOptimizeMemory(graphRef(node))
	return GraphNode(flintNode)
}

///////////////
/// Tensor Operations
//////////////

func Add[T Numeric | GraphNode](operand1 GraphNode, operand2 T) GraphNode {
	var flintNode *C.FGraphNode

	switch c := any(operand2).(type) {
	case int32:
		flintNode = C.fadd_ci(graphRef(operand1), C.int(c))
	case int64:
		flintNode = C.fadd_cl(graphRef(operand1), C.long(c))
	case float32:
		flintNode = C.fadd_cf(graphRef(operand1), C.float(c))
	case float64:
		flintNode = C.fadd_cd(graphRef(operand1), C.double(c))
	case GraphNode:
		flintNode = C.fadd_g(graphRef(operand1), graphRef(c))
	default:
		panic("invalid type")
	}

	return GraphNode(flintNode)
}

func Pow[T Numeric | GraphNode](base GraphNode, exponent T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch c := any(exponent).(type) {
	case GraphNode:
		flintNode = C.fpow_g(graphRef(base), graphRef(c))
	case int32:
		flintNode = C.fpow_ci(graphRef(base), C.int(c))
	case int64:
		flintNode = C.fpow_cl(graphRef(base), C.long(c))
	case float32:
		flintNode = C.fpow_cf(graphRef(base), C.float(c))
	case float64:
		flintNode = C.fpow_cd(graphRef(base), C.double(c))
	default:
		panic("invalid type")
	}
	return GraphNode(flintNode)
}

func Mul[T Numeric | GraphNode](operand1 GraphNode, operand2 T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch c := any(operand2).(type) {
	case GraphNode:
		flintNode = C.fmul_g(graphRef(operand1), graphRef(c))
	case int32:
		flintNode = C.fmul_ci(graphRef(operand1), C.int(c))
	case int64:
		flintNode = C.fmul_cl(graphRef(operand1), C.long(c))
	case float32:
		flintNode = C.fmul_cf(graphRef(operand1), C.float(c))
	case float64:
		flintNode = C.fmul_cd(graphRef(operand1), C.double(c))
	default:
		panic("invalid type")
	}
	return GraphNode(flintNode)
}

// Div Divides numerator by the denominator.
// At least one of the parameters HAS to be a GraphNode
// Division with Tensors is carried out element-wise
func Div[T Numeric | GraphNode](numerator T, denominator T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch x := any(numerator).(type) {
	case GraphNode:
		switch y := any(denominator).(type) {
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
		if y, isNode := any(denominator).(GraphNode); isNode == true {
			flintNode = C.fdiv_ici(C.int(x), graphRef(y))
		} else {
			panic("invalid type")
		}
	case int64:
		if y, isNode := any(denominator).(GraphNode); isNode == true {
			flintNode = C.fdiv_icl(C.long(x), graphRef(y))
		} else {
			panic("invalid type")
		}
	case float32:
		if y, isNode := any(denominator).(GraphNode); isNode == true {
			flintNode = C.fdiv_icf(C.float(x), graphRef(y))
		} else {
			panic("invalid type")
		}
	case float64:
		if y, isNode := any(denominator).(GraphNode); isNode == true {
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
func Sub[T Numeric | GraphNode](minuend T, subtrahend T) GraphNode {
	var flintNode *C.FGraphNode = nil
	switch x := any(minuend).(type) {
	case GraphNode:
		switch y := any(subtrahend).(type) {
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
		if y, isNode := any(subtrahend).(GraphNode); isNode == true {
			flintNode = C.fsub_ici(C.int(x), graphRef(y))
		} else {
			panic("invalid type")
		}
	case int64:
		if y, isNode := any(subtrahend).(GraphNode); isNode == true {
			flintNode = C.fsub_icl(C.long(x), graphRef(y))
		} else {
			panic("invalid type")
		}
	case float32:
		if y, isNode := any(subtrahend).(GraphNode); isNode == true {
			flintNode = C.fsub_icf(C.float(x), graphRef(y))
		} else {
			panic("invalid type")
		}
	case float64:
		if y, isNode := any(subtrahend).(GraphNode); isNode == true {
			flintNode = C.fsub_icd(C.double(x), graphRef(y))
		} else {
			panic("invalid type")
		}
	default:
		panic("invalid type")
	}
	return GraphNode(flintNode)
}

// Log takes the element wise logarithm naturalis of x.
func Log(x GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.flog(graphRef(x))
	return GraphNode(flintNode)
}

// Log2 takes the element wise base 10 logarithm of x.
func Log2(x GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.flog2(graphRef(x))
	return GraphNode(flintNode)
}

// Log10 takes the element wise base 10 logarithm of x.
func Log10(x GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.flog10(graphRef(x))
	return GraphNode(flintNode)
}

// Sin takes the element wise sinus of x.
func Sin(x GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fsin(graphRef(x))
	return GraphNode(flintNode)
}

// Sqrt takes the element wise square root of x.
func Sqrt(x GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fsqrt_g(graphRef(x))
	return GraphNode(flintNode)
}

// Exp takes the each element as the exponent to e.
func Exp(x GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fexp(graphRef(x))
	return GraphNode(flintNode)
}

// Cos takes the element wise cosine of x.
func Cos(x GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fcos(graphRef(x))
	return GraphNode(flintNode)
}

// Tan takes the element wise tangent of x.
func Tan(x GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.ftan(graphRef(x))
	return GraphNode(flintNode)
}

// Asin takes the element wise inverse sinus of x.
func Asin(x GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fasin(graphRef(x))
	return GraphNode(flintNode)
}

// Acos takes the element wise inverse cosinus of x.
func Acos(x GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.facos(graphRef(x))
	return GraphNode(flintNode)
}

// Atan takes the element wise inverse tangent of x.
func Atan(x GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fatan(graphRef(x))
	return GraphNode(flintNode)
}

// Neg swaps the sign of each element.
func Neg(x GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fneg(graphRef(x))
	return GraphNode(flintNode)
}

// Sign applies the sign function to each element.
// i.e. x[i] = 1 if x[i] >= 0 else x[i] = -1
// The input tensor x must have an integer type.
// This function returns a F_INT32 tensor.
func Sign(x GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fsign(graphRef(x))
	return GraphNode(flintNode)
}

// Even gives the result of module 2 for each element.
// i.e. x[i] = 1 if x[i] mod 2 == 0 else x[i] = 0
// This function returns a F_INT32 tensor.
func Even(x GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.feven(graphRef(x))
	return GraphNode(flintNode)
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
	return GraphNode(flintNode)
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
	return GraphNode(flintNode)
}

func Matmul(a GraphNode, b GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fmatmul(graphRef(a), graphRef(b))
	return GraphNode(flintNode)
}

func Flatten(a GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fflatten(graphRef(a))
	return GraphNode(flintNode)
}

func FlattenDim(a GraphNode, dim int) GraphNode {
	var flintNode *C.FGraphNode = C.fflatten_dimension(graphRef(a), C.int(dim))
	return GraphNode(flintNode)
}

func Convert(a GraphNode, newType tensorDataType) GraphNode {
	var flintNode *C.FGraphNode = C.fconvert(graphRef(a), C.enum_FType(newType))
	return GraphNode(flintNode)
}

func Reshape(a GraphNode, shape Shape) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)

	var flintNode *C.FGraphNode = C.freshape(graphRef(a), &(newShape[0]), C.int(len(shape)))
	return GraphNode(flintNode)
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
	return GraphNode(flintNode)
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
	var flintNode *C.FGraphNode = C.freduce_sum(graphRef(a), C.int(dim))
	return GraphNode(flintNode)
}

func ReduceMul(a GraphNode, dim int) GraphNode {
	var flintNode *C.FGraphNode = C.freduce_mul(graphRef(a), C.int(dim))
	return GraphNode(flintNode)
}

func ReduceMin(a GraphNode, dim int) GraphNode {
	var flintNode *C.FGraphNode = C.freduce_min(graphRef(a), C.int(dim))
	return GraphNode(flintNode)
}

func ReduceMax(a GraphNode, dim int) GraphNode {
	var flintNode *C.FGraphNode = C.freduce_max(graphRef(a), C.int(dim))
	return GraphNode(flintNode)
}

func Slice(a GraphNode, start Axes, end Axes) GraphNode {
	newStart := convertArray[uint, C.long](start)
	newEnd := convertArray[uint, C.long](end)

	var flintNode *C.FGraphNode = C.fslice(graphRef(a), &(newStart[0]), &(newEnd[0]))
	return GraphNode(flintNode)
}

func SliceWithStride(a GraphNode, start Axes, end Axes, stride Stride) GraphNode {
	newStart := convertArray[uint, C.long](start)
	newEnd := convertArray[uint, C.long](end)
	newStride := convertArray[int, C.long](stride)

	var flintNode *C.FGraphNode = C.fslice_step(graphRef(a), &(newStart[0]), &(newEnd[0]), &(newStride[0]))
	return GraphNode(flintNode)
}

func Extend(a GraphNode, shape Shape, insertAt Axes) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)
	newInsertAt := convertArray[uint, C.size_t](insertAt)

	var flintNode *C.FGraphNode = C.fextend(graphRef(a), &(newShape[0]), &(newInsertAt[0]))
	return GraphNode(flintNode)
}

func ExtendWithStride(a GraphNode, shape Shape, insertAt Axes, stride Stride) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)
	newInsertAt := convertArray[uint, C.size_t](insertAt)
	newStride := convertArray[int, C.long](stride)

	var flintNode *C.FGraphNode = C.fextend_step(graphRef(a), &(newShape[0]), &(newInsertAt[0]), &(newStride[0]))
	return GraphNode(flintNode)
}

func Concat(a GraphNode, b GraphNode, axis uint) GraphNode {
	var flintNode *C.FGraphNode = C.fconcat(graphRef(a), graphRef(b), C.uint(axis))
	return GraphNode(flintNode)
}

func Expand(a GraphNode, axis uint, size uint) GraphNode {
	var flintNode *C.FGraphNode = C.fexpand(graphRef(a), C.uint(axis), C.uint(size))
	return GraphNode(flintNode)
}

func Abs(a GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fabs_g(graphRef(a))
	return GraphNode(flintNode)
}

func Repeat(a GraphNode, repetitions Axes) GraphNode {
	newRepetitions := convertArray[uint, C.int](repetitions)

	var flintNode *C.FGraphNode = C.frepeat(graphRef(a), &(newRepetitions[0]))
	return GraphNode(flintNode)
}

func Transpose(a GraphNode, axes Axes) GraphNode {
	newAxes := convertArray[uint, C.int](axes)

	var flintNode *C.FGraphNode = C.ftranspose(graphRef(a), &(newAxes[0]))
	return GraphNode(flintNode)
}

func Convolve(a GraphNode, kernel GraphNode, stride Stride) GraphNode {
	newStride := convertArray[int, C.uint](stride)

	var flintNode *C.FGraphNode = C.fconvolve(graphRef(a), graphRef(kernel), &(newStride[0]))
	return GraphNode(flintNode)
}

func Slide(a GraphNode, kernel GraphNode, stride Stride) GraphNode {
	newStride := convertArray[int, C.uint](stride)

	var flintNode *C.FGraphNode = C.fslide(graphRef(a), graphRef(kernel), &(newStride[0]))
	return GraphNode(flintNode)
}

func Index(a GraphNode, indices GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.findex(graphRef(a), graphRef(indices))
	return GraphNode(flintNode)
}

func IndexSet(a GraphNode, b GraphNode, indices GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.findex_set(graphRef(a), graphRef(b), graphRef(indices))
	return GraphNode(flintNode)
}

func SlidingWindow(a GraphNode, size Shape, stride Stride) GraphNode {
	newSize := convertArray[uint, C.size_t](size)
	newStride := convertArray[int, C.uint](stride)

	var flintNode *C.FGraphNode = C.fsliding_window(graphRef(a), &(newSize[0]), &(newStride[0]))
	return GraphNode(flintNode)
}

func Permute(a GraphNode, axis uint) GraphNode {
	var flintNode *C.FGraphNode = C.fpermutate(graphRef(a), C.uint(axis))
	return GraphNode(flintNode)
}
