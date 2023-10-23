/*
Package tensor serves as a wrapper around the specific tensor library.
It abstracts away all technicalities like memory management and error handling.

This package should be used to handle all tensor operations instead of relying on the underlying Library alone and risking memory issues.
*/
package tensor

import "C"
import (
	"github.com/Frobeniusnorm/Flint/go/flint"
	"runtime"
)

type Shape = flint.Shape // this way we keep the functions etc.

type Axes = flint.Axes

// Numeric is a constraint union collection of valid numeric types for use in tensors
// TODO: add uint64? Issues with flint!
type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~float32 | ~float64
}

/*
Tensor essentially acts as a memory-safe wrapper for [flint.GraphNode].
Since go does not have a destructor for GraphNode any references to nodes in C memory cannot be cleaned up.
By having a [Close] method, the memory cleanup can be deferred.
*/
type Tensor struct {
	node *flint.GraphNode
	data *any
	err  error
}

func (x *Tensor) String() string {
	return "Tensor (shape): [data]" // TODO
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
func (x *Tensor) init() {
	if x.node != nil {
		flint.SetRefCounter(*x.node, 1)
	}

	runtime.SetFinalizer(x, func(x *Tensor) { x.Close() })
	runtime.KeepAlive(x)
}

/*
Close resets the ref counter to 0.
This fixes memory management issues, as the node can be cleared on subsequent calls to [flint.OptimizeMemory]
*/
func (x *Tensor) Close() {
	runtime.SetFinalizer(x, nil) // remove any finalizer for this tensor
	if x.node != nil {
		flint.SetRefCounter(*x.node, 0)
	}
	runtime.KeepAlive(x)
}

func (x *Tensor) Shape() Shape {
	return x.node.GetShape()
}

// Value returns the data of a tensor.
func (x *Tensor) Value() any {
	return nil
}

// initNode makes sure that a node is initialized.
// This function should be called when it is required that a data value (usually from scalar)
// needs to be passed as a node.
func (x *Tensor) initNode() {
	if x.node != nil {
		return
	}

	flintNode, err := flint.CreateScalar(x.data)
	if err != nil {
		panic(err)
	}
	x.data = nil
	x.node = &flintNode
}

// TODO: can i, or should i use method decorators?
