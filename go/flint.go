// Package flint contains all types and function declarations similar to [../flint.h]
package flint

/*
#cgo LDFLAGS: -lflint -lOpenCL
#include "../flint.h"
*/
import "C"

type backend int

const (
	BACKEND_ONLY_CPU backend = iota
	BACKEND_ONLY_GPU
	BACKEND_BOTH
)

func Init(backend backend) {
	C.flintInit(C.int(backend))
}

func Cleanup() {}

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

func SetLoggingLevel(level loggingLevel) {}

func Log(level loggingLevel, message string) {}

type floatingPointType int

// FIXME: use a generic interface instead https://en.wikipedia.org/wiki/Go_(programming_language)#Generic_code_using_parameterized_types
const (
	INT32 floatingPointType = iota
	INT64
	FLOAT32
	FLOAT64
)
