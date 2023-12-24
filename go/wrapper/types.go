/*
Package wrapper contains all basic types and functions
as it is only the low-level wrapper, declarations similar to [../wrapper.h]

This wrapper is features complete, but should only be used as a basis for higher level software,
as there is no memory management or similar things.

NOTE: important rules for writing robust CGo code:
- only pass C types to C!
- don't pass data from Go's stack memory to C. include stdlib and use C.malloc!
- only pass unsafe.Pointer and C pointer to cgo!
*/
package wrapper

/*
#cgo LDFLAGS: -lflint -lOpenCL -lstdc++ -lm
#include <flint/flint.h>
#include <stdlib.h> // needed for C.free!
#include <errno.h>

// typedef to get size of pointer using CGo
typedef FGraphNode* graph_ref;
*/
import "C"
import (
	"fmt"
)

// TODO: use go:generate with for the docs (combination with my custom doc generation script)!

///////////////
// Types and Structs
//////////////

type GraphNode struct {
	ref *C.FGraphNode
}

// Valid tells if a GraphNode has a valid structure.
// This can be useful, as a precondition to decide whether to check errno or not.
// BUG: FIXME for now this is not possible as wrapper does not return "invalid" GraphNodes yet.
func (node GraphNode) Valid() bool {
	return true
}

type Result[T completeNumeric] struct {
	resultRef *C.FResultData
	nodeRef   *C.FGraphNode
	Data      ResultData[T]
	Shape     Shape
	DataType  DataType
}

type ResultData[T completeNumeric] []T

func (a ResultData[T]) String() string {
	return fmt.Sprintf("%v", []T(a))
}

// Stride defines the steps for sliding operations
// needs to have one entry for each dimension of flint
type Stride []int

func (a Stride) String() string {
	return fmt.Sprintf("%v", []int(a))
}

// Axes indicate changes in dimensions (i.e. transpose)
// needs to have one entry for each dimension of flint
// and each entry should not be higher than the number of dimensions
type Axes []uint

func (a Axes) String() string {
	return fmt.Sprintf("%v", []uint(a))
}

// Shape represents the size of a flint.
// needs to have one entry for each dimension of flint
type Shape []uint

func (a Shape) NumItems() uint {
	sum := uint(1)
	for _, val := range a {
		sum *= val
	}
	return sum
}

func (a Shape) String() string {
	return fmt.Sprintf("%v", []uint(a))
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

// DataType represents the valid datatypes for the wrapper backend
type DataType uint32 // equal type so it is comparable

// the order is important as it defines the type coercion
const (
	F_INT32 DataType = iota
	F_INT64
	F_FLOAT32
	F_FLOAT64
)

func (x DataType) String() string {
	switch x {
	case F_INT32:
		return "int32"
	case F_INT64:
		return "int64"
	case F_FLOAT32:
		return "float32"
	case F_FLOAT64:
		return "float64"
	default:
		panic("invalid type")
	}
}

//////////////////
// Number types
//////////////////

// baseNumeric is a constraint interface representing the supported types of C operations
type baseNumeric interface {
	~int32 | ~int64 | ~float32 | ~float64
}

// completeNumeric is a constraint interface representing all numeric types that can be used in wrapper!
// NOTE: cant support uint64 as casting it to int64 might cause overflows
type completeNumeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~float32 | ~float64
}

// cNumbers represents the basic C types available in CGo
// NOTE: size_t should be equivalent to ulong
type cNumbers interface {
	C.int | C.size_t | C.long | C.uint | C.float | C.double
}
