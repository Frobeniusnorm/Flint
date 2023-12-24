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

// TODO: use go:generate with for the docs (combination with my custom doc generation script)!

///////////////
// Types and Structs
//////////////

type GraphNode struct {
	ref *C.FGraphNode
	// ref *graphNode
}

type operation struct {
	shape             *C.size_t
	additional_data   unsafe.Pointer
	op_type           C.enum_FOperationType
	data_type         C.enum_FType
	dimensions        C.int
	broadcasting_mode C.int
}

type resultData struct {
	data        unsafe.Pointer
	num_entries C.size_t
}

type graphNode struct {
	num_predecessor   C.int
	predecessor       **C.FGraphNode
	operation         operation
	reference_counter C.size_t
	result_data       *resultData
	gradient_data     unsafe.Pointer
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

// DataType represents the valid datatypes for the wrapper backend
type DataType uint32 // equal type so it is comparable

// the order is important as it defines the type coercion
const (
	F_INT32 DataType = iota
	F_INT64
	F_FLOAT32
	F_FLOAT64
	F_GRAPH
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
