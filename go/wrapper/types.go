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
	"unsafe"
)

///////////////
// Types and Structs
//////////////

type GraphNode struct {
	ref *C.FGraphNode
	// ref *graphNode
}

type Int = int32
type Long = int64
type Float = float32
type Double = float64

///////////
// Extraction Functions
///////////

// GetResultValue fetches the result data from the C backend and returns it as a Go type
// NOTE: this will not execute the graph or change anything in memory
// So make sure to execute the graph before calling this function!
func GetResultValue[T Numeric](node GraphNode) []T {
	SyncMemory(node) // make sure data is accessible on CPU

	var flintNode *C.FGraphNode = node.ref

	dataSize := int(flintNode.result_data.num_entries)
	dataPtr := unsafe.Pointer(flintNode.result_data.data)
	dataType := DataType(flintNode.operation.data_type)

	return fromCArrayToGo[T](dataPtr, dataSize, dataType)
}

// GetShape returns the shape of a graph node
// NOTE: this will not execute the graph or change anything in memory
func (node GraphNode) GetShape() Shape {
	var flintNode *C.FGraphNode = node.ref
	shapePtr := unsafe.Pointer(flintNode.operation.shape)
	shapeSize := int(flintNode.operation.dimensions)
	return fromCArrayToGo[uint](shapePtr, shapeSize, INT64) // as the C backend stores the shape as int64
}

func (node GraphNode) GetType() DataType {
	var flintNode *C.FGraphNode = node.ref
	return DataType(flintNode.operation.data_type)
}

type ResultData[T completeNumeric] []T

func (a ResultData[T]) String() string {
	return fmt.Sprintf("%v", []T(a))
}

//////////////////
// NUMERIC TYPES
//////////////////

// DataType represents FType -- the valid datatypes for the wrapper backend
type DataType uint32 // equal type so it is comparable

// the order is important as it defines the type coercion
const (
	INT32 DataType = iota
	INT64
	FLOAT32
	FLOAT64
)

func (x DataType) String() string {
	switch x {
	case INT32:
		return "int32"
	case INT64:
		return "int64"
	case FLOAT32:
		return "float32"
	case FLOAT64:
		return "float64"
	default:
		panic("invalid type")
	}
}

// Numeric represents the numeric types that can be used in wrapper!
type Numeric interface {
	~Int | ~Long | ~Float | ~Double
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

//////////////////
// SHAPE
//////////////////

// Shape represents the size of a flint.
// needs to have one entry for each dimension of flint
type Shape []uint

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

func (a Shape) NumElements() uint {
	var result uint = 1
	for _, v := range a {
		result *= v
	}
	return result
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
type Axes []int

func (a Axes) String() string {
	return fmt.Sprintf("%v", []int(a))
}
