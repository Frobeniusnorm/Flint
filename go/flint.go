// Package flint contains all types and function declarations similar to [../flint.h]
package flint

/*
#cgo CXXFLAGS: -std=c++23 -I.. -flto
#cgo LDFLAGS: -lflint -lOpenCL -lstdc++ -L.. -lOpenCL
#include "../flint.h"
*/
import "C"
import "fmt"

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
	fmt.Println(new(C.enum_FLogType))
	//C.flogging()
}

func Log(level loggingLevel, message string) {}

type floatingPointType int

// FIXME: use a generic interface instead https://en.wikipedia.org/wiki/Go_(programming_language)#Generic_code_using_parameterized_types
const (
	INT32 floatingPointType = iota
	INT64
	FLOAT32
	FLOAT64
)
