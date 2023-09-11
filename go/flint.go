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
	return bool(C.fIsEagerExecution())
}

type floatingPointType int

// FIXME: use a generic interface instead https://en.wikipedia.org/wiki/Go_(programming_language)#Generic_code_using_parameterized_types
// FIXME: or a map from go type to c type?
const (
	INT32 floatingPointType = iota
	INT64
	FLOAT32
	FLOAT64
)

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

type Shape []int

type Operation struct {
	dimension int
	shape     Shape
	opType    operationType
	fpType    floatingPointType
	extraData []byte
}

type GraphNode struct {
	predecessors []GraphNode
	operation    Operation
	// TODO: result data
	// TODO: gradient data
}
