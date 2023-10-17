/*
Package flint contains all basic types and functions
as it is only the low-level wrapper, declarations similar to [../flint.h]

This wrapper is features complete, but should only be used as a basis for higher level software,
as there is no memory management or similar things.

NOTE: important rules for writing robust CGo code:
- only pass C types to C!
- don't pass data from Go's stack memory to C. include stdlib and use C.malloc!
- only pass unsafe.Pointer and C pointer to cgo!
*/
package flint

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

///////////////
// Types and Structs
//////////////

type GraphNode struct {
	ref *C.FGraphNode
}

// Valid tells if a GraphNode has a valid structure.
// This can be useful, as a precondition to decide whether to check errno or not.
// BUG: FIXME for now this is not possible as flint does not return "invalid" GraphNodes yet.
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
// needs to have one entry for each dimension of tensor
type Stride []int

func (a Stride) String() string {
	return fmt.Sprintf("%v", []int(a))
}

// Axes indicate changes in dimensions (i.e. transpose)
// needs to have one entry for each dimension of tensor
// and each entry should not be higher than the number of dimensions
type Axes []uint

func (a Axes) String() string {
	return fmt.Sprintf("%v", []uint(a))
}

// Shape represents the size of a tensor.
// needs to have one entry for each dimension of tensor
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

// DataType represents the valid datatypes for the flint backend
type DataType uint32

const (
	f_INT32 DataType = iota
	f_INT64
	f_FLOAT32
	f_FLOAT64
)

//////////////////
// Number types
//////////////////

// baseNumeric is a constraint interface representing the supported types of C operations
type baseNumeric interface {
	~int32 | ~int64 | ~float32 | ~float64
}

// completeNumeric is a constraint interface representing all numeric types that can be used in flint!
// NOTE: cant support uint64 as casting it to int64 might cause overflows
type completeNumeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~float32 | ~float64
}

// cNumbers represents the basic C types available in CGo
// NOTE: size_t should be equivalent to ulong
type cNumbers interface {
	C.int | C.size_t | C.long | C.uint | C.float | C.double
}
