/*
Package tensor serves as a wrapper around the specific tensor library.
It abstracts away all technicalities like memory management and error handling.

This package should be used to handle all tensor operations instead of relying on the underlying Library alone and risking memory issues.
*/
package tensor

import "C"
import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"reflect"
	"runtime"
)

type Shape = flint.Shape // this way we keep the functions etc.

type Axes = flint.Axes

// Numeric is a constraint union collection of valid numeric types for use in tensors
// TODO: add uint64 and uintptr? Issues with flint!
type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~float32 | ~float64
}

/*
Tensor essentially acts as a memory-safe wrapper for [flint.GraphNode].
Since go does not have a destructor for GraphNode any references to nodes in C memory cannot be cleaned up.
By having a [Close] method, the memory cleanup can be deferred.
*/
type Tensor struct {
	node     *flint.GraphNode
	err      error
	data     *any         // only in case the actual type doesn't matter. Else use one of the below
	dataType reflect.Kind // FIXME: or type?
	// This is the horrible workaround to keep this structure free of reflection and generics ...
	data_int     *int
	data_int8    *int8
	data_int16   *int16
	data_int32   *int32
	data_int64   *int64
	data_uint    *uint
	data_uint8   *uint8
	data_uint16  *uint16
	data_uint32  *uint32
	data_float32 *float32
	data_float64 *float64
}

// String prints the contents of the tensor.
// NOTE: If not a light tensor, this causes the node's graph to be executed!
// Use carefully! (best only in debugging)
func (x Tensor) String() string {
	if x.node == nil {
		return fmt.Sprintf("Tensor (1): [%v]", x.data)
	} else {
		data, err := flint.CalculateResult[int](*x.node)
		if err != nil {
			panic(err)
		}
		return fmt.Sprintf("Tensor (%s): [%v]", x.node.GetShape(), data)
	}
}

/*
init is an internal method to initialize a Tensor struct.
This means setting the reference counter to 1 on flint nodes.
This fixes memory management issues, as the node won't be cleared on subsequent calls to [flint.OptimizeMemory]

Experimentally we currently use a [finalizer] in this method. This means calling the close method is not required!
However, this may change in the future as we have a massive [finalizer overhead]

[finalizer]: https://medium.com/a-journey-with-go/go-finalizers-786df8e17687
[finalizer overhead]: https://lemire.me/blog/2023/05/19/the-absurd-cost-of-finalizers-in-go/

[finalizer example]: https://gist.github.com/deltamobile/6511901
*/
func (x Tensor) init() {
	if x.node != nil {
		flint.SetRefCounter(*x.node, 1)
	}

	// FIXME: pointer or not in finalizer parameter?
	runtime.SetFinalizer(x, func(x *Tensor) { x.Close() })
	runtime.KeepAlive(x)
}

/*
Close resets the ref counter to 0.
This fixes memory management issues, as the node can be cleared on subsequent calls to [flint.OptimizeMemory]
*/
func (x Tensor) Close() {
	runtime.SetFinalizer(x, nil) // remove any finalizer for this tensor
	if x.node != nil {
		flint.SetRefCounter(*x.node, 0)
	}
	runtime.KeepAlive(x)
}

func (x Tensor) Shape() Shape {
	if x.node == nil {
		return Shape{1}
	} else {
		return x.node.GetShape()
	}
}

// Value returns the data of a tensor.
func (x Tensor) Value() any {
	return nil // TODO
}

// node makes sure that a node is initialized and returns it.
// This function should be called when it is required that a data value (usually from scalar)
// needs to be passed as a node.
// FIXME: isn't this useless without a pointer receiver?
func (x Tensor) readyNode() Tensor {
	if x.node != nil {
		return x // no changes needed
	}

	reflect.ValueOf(x.data).Elem().Type().
		flintNode, err := flint.CreateScalar(*x.data)
	if err != nil {
		panic(err)
	}
	x.data = nil
	x.node = &flintNode
	return x
}

// val returns a possible value.
// This is either the node or the value from a light tensor
func (x Tensor) val() any {
	if x.node != nil {
		return *x.node
	} else if x.data != nil {
		return *x.data
	} else {
		panic("no value present - invalid tensor?")
	}
}

func (x Tensor) light() bool {
	return x.node != nil
}

func readyTensor(x Tensor, y Tensor) (a Tensor, b Tensor) {
	if !x.light() {
		return x, y
	} else if !y.light() {
		return y, x
	} else {
		return x.readyNode(), y
	}
}

// TODO: can i, or should i use method decorators?
