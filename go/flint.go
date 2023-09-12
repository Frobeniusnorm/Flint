// Package flint contains all types and function declarations similar to [../flint.h]
package flint

/*
#cgo CXXFLAGS: -std=c++23 -I.. -flto
#cgo LDFLAGS: -lflint -lOpenCL -lstdc++ -L.. -lOpenCL
#include "../flint.h"
*/
import "C"
import (
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
	// FIXME: this might break when a new backend is added
	return Backend(C.flintInitializedBackends())
}

func Cleanup() {
	C.flintCleanup()
}

type loggingLevel int

// FIXME: or use Undefined logging level?
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

func SetEagerExecution(on bool) {
	if on == true {
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

type tensorDataType interface {
	~int32 | ~int64 | ~float32 | ~float64
}

func CreateGraph[T tensorDataType](data []T, shape Shape) GraphNode[T] {
	dataPointer := unsafe.Pointer(&data)
	shapePointer := unsafe.Pointer(&shape)
	shapePointer = (*C.ulong)(shapePointer)

	C.fCreateGraph(dataPointer, C.int(len(data)), C.F_FLOAT32, shapePointer, C.int(len(shape)))

	return GraphNode[T]{}
}
