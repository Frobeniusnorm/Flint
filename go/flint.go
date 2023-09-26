// Package flint contains all basic types and functions
// as it is only the low-level wrapper, declarations similar to [../flint.h]
//
// this wrapper is features complete, but should only be used as a basis for higher level software,
// as there is no memory management or similar things.
package flint

/*
NOTE: important rules for writing robust CGo code:
- only pass C types to C!
- don't pass data from Go's stack memory to C. include stdlib and use C.malloc!
- only pass unsafe.Pointer and C pointer to cgo!
*/

/*
#cgo LDFLAGS: -lflint -lOpenCL -lstdc++ -lm
#include <flint/flint.h>
#include <stdlib.h>  // needed for C.free!
#include <errno.h>

// simple go function for the go interface to reset the errno
// WARNING: this MAY break in multi-threaded scenarios
void reset_errno(void) {
	errno = 0;
}

// typedef to get size of pointer using CGo
typedef  FGraphNode* graph_ref;
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
	Data  []T
	Shape Shape
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

// The ResultData struct holds a pointer to some C memory including the output of an evaluated node
type ResultData unsafe.Pointer

func resultRef(res ResultData) *C.FResultData {
	return (*C.FResultData)(res)
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

func (a Shape) NumItems() uint {
	sum := uint(0)
	for _, val := range a {
		sum += val
	}
	return sum
}

func (a Shape) Equal(b Shape) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

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

type cTypes interface {
	C.FGraphNode | *C.FGraphNode
}

// convertArray converts Arrays between arbitrary types
// NOTE: does not work with nested arrays yet.
// See: https://stackoverflow.com/questions/71587996/cannot-use-type-assertion-on-type-parameter-value
// FIXME: func convertArray[In any, Out any](arr []In) []Out {
func convertArray[In completeNumbers, Out completeNumbers | cNumbers](arr []In) []Out {
	result := make([]Out, len(arr))
	for idx, val := range arr {
		result[idx] = Out(val)
		//x := any(val).(Out)
		//result[idx] = x
	}
	return result
}

//////////////
// SETUP
/////////////

// Error offers a generic error struct in flint
type Error struct {
	message string
}

// FIXME: this stuff is currently unused
func (err *Error) Error() string {
	return err.message
}

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

func CleanupCpu() {
	C.flintCleanup_cpu()
}

func CleanupGpu() {
	C.flintCleanup_gpu()
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

// LoadImage turns an image into a 3 dimensional tensor
// NOTE: this function exemplary shows how to properly deal with errno
func LoadImage(path string) (GraphNode, error) {
	C.reset_errno()

	unsafePath := C.CString(path)
	defer C.free(unsafe.Pointer(unsafePath))

	var flintNode *C.FGraphNode
	var err error // has underlying syscall.Errno type
	flintNode, err = C.fload_image(unsafePath)
	if err != nil {
		return nil, err
	}
	return GraphNode(flintNode), nil
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

func CreateGraph[T completeNumbers](data []T, shape Shape) GraphNode {
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

/*
SyncMemory flushes all GPU data to the CPU.
fExecuteGraph does not guarantee that memory is present on the cpu (it may
be kept on the GPU for performance reasons). This method enforces all GPU
data to be flushed to the CPU (but never executes the node!).
Also see fCalculateResult.
*/
func SyncMemory(node GraphNode) ResultData {
	res := C.fSyncMemory(graphRef(node))
	return ResultData(res)
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
		Data:  result,
		Shape: shape,
	}
}

/*
CalculateGradient Calculates the overall gradient of an output node to a variable.
The variable must be marked as a gradient variable, see
fMarkGradientVariable and the output node node must be constructed
in a gradient context (to remember which variables are directly or
indirectly present in which operation), which can be started with
fStartGradientContext. If you need to compute multiple gradients for one
output use fCalculateGradients since it is far more efficient.
  - node: the Node which represents the chain of functions of which
    the gradient is to be computed.
  - dx: the variable for which node is derived for
*/
func CalculateGradient(node GraphNode, dx GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fCalculateGradient(graphRef(node), graphRef(dx))
	return GraphNode(flintNode)
}

/*
CalculateGradients Calculates the overall gradient of an output node to multiple variables.
The variables must be marked as a gradient variable, see
fMarkGradientVariable and the output node node must be constructed
in a gradient context (to remember which variables are directly or
indirectly present in which operation), which can be started with
fStartGradientContext.
  - node: the Node which represents the chain of functions of which
    the gradients are to be computed.
  - dxs: array of variables for which node is derived for.

returns array with the same size as dxs
*/
func CalculateGradients(node GraphNode, dxs []GraphNode) []GraphNode {
	//partials := convertArray[GraphNode, *C.FGraphNode](dxs)
	//resPtr := C.malloc(C.ulong(len(partials) * C.sizeof_graph_ref))
	//C.fCalculateGradients(graphRef(node), &(partials[0]), C.uint(len(partials)), (**C.FGraphNode)(resPtr))
	// FIXME: idk
	// convertArray[*C.FGraphNode, GraphNode](resPtr)
	return nil
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

func ExecuteGraphCpu(node GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fExecuteGraph_cpu(graphRef(node))
	return GraphNode(flintNode)
}

func ExecuteGraphGpu(node GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fExecuteGraph_gpu(graphRef(node))
	return GraphNode(flintNode)
}

func ExecuteGraphCpuEagerly(node GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fExecuteGraph_cpu_eagerly(graphRef(node))
	return GraphNode(flintNode)
}

func ExecuteGraphGpuEagerly(node GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fExecuteGraph_gpu_eagerly(graphRef(node))
	return GraphNode(flintNode)
}

func OptimizeMemory(node GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fOptimizeMemory(graphRef(node))
	return GraphNode(flintNode)
}

///////////////
/// Tensor Operations
//////////////

/*
Add [operand1] and [operand2].
At least one of the parameters HAS to be a [GraphNode].
Addition is carried out element-wise.
*/
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

/*
Pow takes [base] to the power of [exponent].
The [exponent] can be a constant number or another [GraphNode].
The operation is carried out element-wise.
*/
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

/*
Mul [operand1] by [operand2].
At least one of the parameters HAS to be a [GraphNode]
Multiplication is carried out element-wise.
*/
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

/*
Divide [numerator] by [denominator].
At least one of the parameters HAS to be a [GraphNode]
Division is carried out element-wise.
*/
func Divide[T Numeric | GraphNode](numerator T, denominator T) GraphNode {
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

/*
Sub Subtracts [minuend] by [subtrahend].
At least one of the parameters HAS to be a [GraphNode].
Subtraction is carried out element-wise.
*/
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

// Exp takes each element as the exponent for power of base e.
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

// Acos takes the element wise inverse cosine of x.
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

/*
Sign applies the sign function to each element.
i.e. x[i] = 1 if x[i] >= 0 else x[i] = -1
The input tensor [x] must have an integer type.
This function returns a [F_INT32] [GraphNode].
*/
func Sign(x GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fsign(graphRef(x))
	return GraphNode(flintNode)
}

/*
Even gives the result of module 2 for each element.
i.e. x[i] = 1 if x[i] mod 2 == 0 else x[i] = 0
This function returns a [F_INT32] [GraphNode].
*/
func Even(x GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.feven(graphRef(x))
	return GraphNode(flintNode)
}

/*
Equal compares a tensor and a constant elementwise by [a] = [b] and returns a 0,1 [F_INT32] [GraphNode].
*/
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

/*
Greater compares a tensor and a constant elementwise by [a] > [b] and returns a 0,1 [F_INT32] [GraphNode].
*/
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

/*
Less compares a tensor and a constant elementwise by [a] < [b] and returns a 0,1 [F_INT32] [GraphNode].
*/
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

/*
Matmul carries out matrix multiplication on the last two dimensions of the tensors.

E.g. a matrix multiplication of two tensors with shapes (64, 32, 16) and (16, 24) will yield a tensor with shape (64, 32, 24).
Since for one entry of the tensor multiple other previous entries are needed, the operand tensors need to be executed first.
Therefor the method will implicitly (or eagerly) execute the two parameter nodes [a] and [b] if their data is not already present.
*/
func Matmul(a GraphNode, b GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fmatmul(graphRef(a), graphRef(b))
	return GraphNode(flintNode)
}

/*
Flatten the complete tensor to a tensor with one dimension.
E.g:

	Flatten([[[3, 1, 4], [2, 1, 5]], [[0, 4, 2], [4, 7, 9]]]) =
		[3, 1, 4, 2, 1, 5, 0, 4, 2, 4, 7, 9]`.
*/
func Flatten(a GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fflatten(graphRef(a))
	return GraphNode(flintNode)
}

/*
FlattenDim flattens a tensor [a] with n dimensions along dimension [dim], resulting in a tensor with n-1 dimensions.
Flattening a dimension will remove it from the shape of the tensor, therefor it's not possible to flatten the dimension 0.

E.g:

	FlattenDim([[[3, 1, 4], [2, 1, 5]], [[0, 4, 2], [4, 7, 9]]], 1) =
		[[3,1,4], [2,1,5], [0,4,2], [4,7,9]]
*/
func FlattenDim(a GraphNode, dim int) GraphNode {
	var flintNode *C.FGraphNode = C.fflatten_dimension(graphRef(a), C.int(dim))
	return GraphNode(flintNode)
}

/*
Convert the data of [a] to the type given by [newType].
*/
func Convert(a GraphNode, newType tensorDataType) GraphNode {
	var flintNode *C.FGraphNode = C.fconvert(graphRef(a), C.enum_FType(newType))
	return GraphNode(flintNode)
}

/*
Reshape the underlying data of the tensor to the new shape.
The product of each dimension of the new shape must be the same as the product of the dimensions of the previous shape.
This means it must describe the same number of entries of the tensor.
*/
func Reshape(a GraphNode, shape Shape) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)

	var flintNode *C.FGraphNode = C.freshape(graphRef(a), &(newShape[0]), C.int(len(shape)))
	return GraphNode(flintNode)
}

/*
Min takes the minimum of two tensors (or a tensor and value) element wise along the last dimension of each.
*/
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

/*
Max takes the maximum of two tensors (or a tensor and value) element wise along the last dimension of each.
*/
func Max[T Numeric | GraphNode](a GraphNode, b T) GraphNode {
	var flintNode *C.FGraphNode = nil
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

/*
ReduceSum reduces one dimension of the tensor by additive folding.

	ReduceSum([[1,2,3], [4,5,6]], 0) = [5,7,9]
	ReduceSum([[1,2,3], [4,5,6]], 1) = [6,15]

The results of the predecessor node must be available, to
ensure that the method may execute the parameter node.
*/
func ReduceSum(a GraphNode, dim int) GraphNode {
	var flintNode *C.FGraphNode = C.freduce_sum(graphRef(a), C.int(dim))
	return GraphNode(flintNode)
}

/*
ReduceMul reduces one dimension of the tensor by multiplicative folding.

E.g:

	ReduceMul([[1,2,3], [4,5,6]], 0) = [4,10,18]
	ReduceMul([[1,2,3], [4,5,6]], 1) = [6, 120]

The results of the predecessor node must be available; to ensure that the method may execute the parameter node.
*/
func ReduceMul(a GraphNode, dim int) GraphNode {
	var flintNode *C.FGraphNode = C.freduce_mul(graphRef(a), C.int(dim))
	return GraphNode(flintNode)
}

/*
ReduceMin reduces one dimension of the tensor by keeping the minimum.

E.g:

	ReduceMin([[1,32,3], [4,5,3]], 0) = [1,5,3]
	ReduceMin([[9,2,3], [-1,5,6]], 1) = [2, -1]

The results of the predecessor node must be available; to ensure that the method may execute the parameter node.
*/
func ReduceMin(a GraphNode, dim int) GraphNode {
	var flintNode *C.FGraphNode = C.freduce_min(graphRef(a), C.int(dim))
	return GraphNode(flintNode)
}

/*
ReduceMax reduces one dimension of the tensor by keeping the maximum.

E.g:

	ReduceMax([[1,32,3], [4,5,3]], 0) = [4,32,3]
	ReduceMax([[9,2,3], [-1,5,6]], 1) = [9, 6]

The results of the predecessor node must be available; to ensure that the method may execute the parameter node.
*/
func ReduceMax(a GraphNode, dim int) GraphNode {
	var flintNode *C.FGraphNode = C.freduce_max(graphRef(a), C.int(dim))
	return GraphNode(flintNode)
}

/*
Slice selects a slice of the tensor with a dimension wise start and end index.
[start] and [end] are arrays with as many entries as the tensor has dimensions.
They may contain negative values, which are then subtracted from the end of the tensor
(e.g. -1 means the element before the last element).
[start] is inclusive and describes the start index of the selection per dimension and [end] describes the end index per dimension and is exclusive.
*/
func Slice(a GraphNode, start Axes, end Axes) GraphNode {
	newStart := convertArray[uint, C.long](start)
	newEnd := convertArray[uint, C.long](end)

	var flintNode *C.FGraphNode = C.fslice(graphRef(a), &(newStart[0]), &(newEnd[0]))
	return GraphNode(flintNode)
}

/*
SliceWithStride selects a slice of the tensor [node] with a dimension wise start index, end index and stride.
[start], [end] and [stride] are arrays with as many entries as the tensor has dimensions.
[start] and [end] may contain negative values, which are then subtracted from the end of the tensor
(e.g. -1 means the element before the last element).
[start] is inclusive and describes the start index of the selection per dimension and [end] describes the end index per dimension and is exclusive.
[stride] contains the per dimension step size (e.g. `2` meaning every second element will be selected etc.) and may be negative as well,
which reverses the traversal order (the first elements are selected as the last ones).
For a negative stride, [start] > [end] must hold (for a positive of course [end] > [start]) for each dimension.
*/
func SliceWithStride(node GraphNode, start Axes, end Axes, stride Stride) GraphNode {
	// TODO: check that axes has the right length compared to node! (or does the C function do this?)
	newStart := convertArray[uint, C.long](start)
	newEnd := convertArray[uint, C.long](end)
	newStride := convertArray[int, C.long](stride)

	var flintNode *C.FGraphNode = C.fslice_step(graphRef(node), &(newStart[0]), &(newEnd[0]), &(newStride[0]))
	return GraphNode(flintNode)
}

/*
Extend creates a new tensor of zeroes with the requested shape.
The original tensor is embedded at the given indices.
  - [node]: original tensor which shape is to be extended
  - [shape]: array of new sizes per dimension. Has the same number of entries and the shape of [node].
  - [insertAt]: array with indices per dimension of [node], denoting where [node] is to be placed in the resulting tensor
*/
func Extend(node GraphNode, shape Shape, insertAt Axes) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)
	newInsertAt := convertArray[uint, C.size_t](insertAt)

	var flintNode *C.FGraphNode = C.fextend(graphRef(node), &(newShape[0]), &(newInsertAt[0]))
	return GraphNode(flintNode)
}

/*
ExtendWithStride creates a new tensor of zeroes with the requested shape.
The original tensor is embedded at the given indices.
  - [node]: original tensor which shape is to be extended,
  - [shape]: array of new sizes per dimension. Has the same number of entries as [node] has dimensions.
  - [insertAt]: array with indices per dimension denoting where [node] is to be placed in the resulting tensor. Has a value per dimension of [node].
  - [stride]: allows to pull apart [node], em-placing zeros between each value of [node]. Has a value per dimension.
*/
func ExtendWithStride(node GraphNode, shape Shape, insertAt Axes, stride Stride) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)
	newInsertAt := convertArray[uint, C.size_t](insertAt)
	newStride := convertArray[int, C.long](stride)

	var flintNode *C.FGraphNode = C.fextend_step(graphRef(node), &(newShape[0]), &(newInsertAt[0]), &(newStride[0]))
	return GraphNode(flintNode)
}

/*
Concat two nodes ([nodeA], [nodeB]) with each other along an [axis].
The nodes have to have the same type and dimensions.

E.g:

	Concat({[[0, 1], [2, 3]], [[4, 5], [6, 7]]}, 0) = [[0, 1], [2, 3], [4, 5], [6, 7]]

	Concat({[[0, 1], [2, 3]], [[4, 5], [6, 7]]}, 1) = [[0, 1, 4, 5], [2, 3, 6, 7]]
*/
func Concat(nodeA GraphNode, nodeB GraphNode, axis uint) GraphNode {
	var flintNode *C.FGraphNode = C.fconcat(graphRef(nodeA), graphRef(nodeB), C.uint(axis))
	return GraphNode(flintNode)
}

/*
Expand adds a new dimension at an arbitrary position to the tensor and repeats the following dimensions to match a given shape.
  - [axis]: the dimension prior to which the new dimension will be inserted (0 means a new dimension in the front, n means as a new last dimension).
  - [size]: the new size of that dimension (repeats the following dimensions ax_size - 1 times).
*/
func Expand(a GraphNode, axis uint, size uint) GraphNode {
	var flintNode *C.FGraphNode = C.fexpand(graphRef(a), C.uint(axis), C.uint(size))
	return GraphNode(flintNode)
}

/*
Abs takes the elementwise absolute value of [node], i.e. |a[i]|
*/
func Abs(node GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fabs_g(graphRef(node))
	return GraphNode(flintNode)
}

/*
Repeat dimensions of a tensor multiple times.
  - [node]: the node in which dimensions are to be repeated
  - [axes]: array with the same number of entries as the tensor has dimensions

E.g:

	Repeat([[0,1], [2,3]], [2, 3]) =
		[[0,1,0,1,0,1], [2,3,2,3,2,3], [0,1,0,1,0,1], [2,3,2,3,2,3]]
*/
func Repeat(node GraphNode, repetitions Axes) GraphNode {
	newRepetitions := convertArray[uint, C.int](repetitions)

	var flintNode *C.FGraphNode = C.frepeat(graphRef(node), &(newRepetitions[0]))
	return GraphNode(flintNode)
}

/*
Transpose tensor [node] along multiple dimensions.
The array [axes] has the same number of entries as [node] has dimensions, which gives the permutations of dimensions.

The tensor will have a resulting shape in which the size each dimension corresponds to the former size in dimension in [axes].
*/
func Transpose(node GraphNode, axes Axes) GraphNode {
	newAxes := convertArray[uint, C.int](axes)

	var flintNode *C.FGraphNode = C.ftranspose(graphRef(node), &(newAxes[0]))
	return GraphNode(flintNode)
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
func Convolve(node GraphNode, kernel GraphNode, stride Stride) GraphNode {
	newStride := convertArray[int, C.uint](stride)

	var flintNode *C.FGraphNode = C.fconvolve(graphRef(node), graphRef(kernel), &(newStride[0]))
	return GraphNode(flintNode)
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
func Slide(node GraphNode, kernel GraphNode, stride Stride) GraphNode {
	newStride := convertArray[int, C.uint](stride)

	var flintNode *C.FGraphNode = C.fslide(graphRef(node), graphRef(kernel), &(newStride[0]))
	return GraphNode(flintNode)
}

/*
Index selects single elements with an index-tensor (integer tensor containing indices for the selected dimension).
It indexes a dimension of the input tensor and the result has the shape of the input tensor except for the indexed dimension.
It is assumed that except for the last entry its shape is a prefix of the shape of the input tensor and the indexing will occur in the matched subsets.
The last dimension of indices is the one indexed in node.

TODO example descriptions:

	Index([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
			[1, 0]) =
		[[[4, 5], [6, 7]], [[0, 1], [2, 3]]]

	Index([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
			[0, 0, 1]) =
		[[[0, 1], [0, 1], [1, 2]], [[3, 4], [3, 4], [5, 6]], [[7, 8], [7, 8], [9, 10]]]

	Index([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
			[[0], [1], [0]]) =
		[[[0], [2]], [[5], [7]], [[8], [10]]]

	Index([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
			[[0, 0], [1, 0], [0, 1]]) =
		[[[0, 0], [2, 2]], [[5, 4], [7, 6]], [[8, 9], [10, 11]]]
*/
func Index(node GraphNode, indices GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.findex(graphRef(node), graphRef(indices))
	return GraphNode(flintNode)
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
func IndexSet(nodeA GraphNode, nodeB GraphNode, indices GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.findex_set(graphRef(nodeA), graphRef(nodeB), graphRef(indices))
	return GraphNode(flintNode)
}

/*
SlidingWindow moves a window view with size along the node by starting
with aligning the first element of the view with the first element of the node,
copying the elements of the view and moving the window by the stride
given for each dimension (the window is first moved in the innermost
dimension and after each is iterated moves it in the outer dimensions).
Each view becomes a new element in a new outer dimension.

Moves a 3x2 rectangle across the node, each time taking one stride in each direction:

	fsliding_window([[0, 1], [2, 3], [4, 5], [6, 7]], [3, 2], [1, 1]) =
		[[[0, 1], [2, 3], [4, 5]], [[2, 3], [4, 5], [6, 7]]]

Moves a 2x2x2 cube across the node, this time moving 2 across the first and last axis for each stride:

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

	var flintNode *C.FGraphNode = C.fsliding_window(graphRef(node), &(newSize[0]), &(newStride[0]))
	return GraphNode(flintNode), nil
}

/*
Permute randomly permutes (= swaps multiple elements with each other without creating, copying or deleting new ones) one axis of the input tensor.
*/
func Permute(a GraphNode, axis uint) (GraphNode, error) {
	var flintNode *C.FGraphNode = C.fpermutate(graphRef(a), C.uint(axis))
	return GraphNode(flintNode), nil
}
