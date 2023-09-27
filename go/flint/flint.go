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
	"syscall"
)

///////////////
// Types and Structs
//////////////

type GraphNode struct {
	ref *C.FGraphNode
}

type ResultData struct {
	ref   *C.FResultData
	Data  any
	Shape Shape
}

// numeric is a constraint interface representing the supported types of C operations
// TODO: as we no longer cast values between C and GO (we shouldn't!) this can be extended!
type numeric interface {
	~int32 | ~int64 | ~float32 | ~float64
}

// Stride defines the steps for sliding operations
// needs to have one entry for each dimension of tensor
type Stride []int

// Axes indicate changes in dimensions (i.e. transpose)
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
	numeric | ~uint | ~int | ~int8 | ~int64 | ~uint64 | ~uint16 | ~uint8
}

// cNumbers represents the usable C types
// size_t is equivalent to ulong
type cNumbers interface {
	C.int | C.size_t | C.long | C.uint | C.float | C.double
}

// cNumbers is a constraint interface for commonly used C Types
type cTypes interface {
	C.FGraphNode | *C.FGraphNode
}

// Error offers a generic error struct in flint
type Error struct {
	message string
	errno   syscall.Errno
}

func (err *Error) Error() string {
	return err.message
}

func resetErrno() {
	C.reset_errno()
}
