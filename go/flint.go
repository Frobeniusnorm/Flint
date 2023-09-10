// Package flint contains all types and function declarations similar to [../flint.h]
package flint

// #include "../flint.h"
import "C"

type backend int

// FIXME: or use iota? Would probably be incompatible with C
const (
	BACKEND_ONLY_CPU backend = iota
	BACKEND_ONLY_GPU
	BACKEND_BOTH
)

func Init(backend backend) {}

func Cleanup() {}

type loggingLevel int

const (
	// FIXME: or use Undefined logging level?
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

const (
	INT32 floatingPointType = iota
	INT64
	FLOAT32
	FLOAT64
)
