/*
Package flint serves as a wrapper around the specific flint library.
It abstracts away all technicalities like memory management and error handling.

This package should be used to handle all flint operations instead of relying on the underlying Library alone and risking memory issues.
*/
package flint

import "C"
import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
	"log"
	"runtime"
)

type Shape = wrapper.Shape // this way we keep the functions etc.

type Axes = wrapper.Axes

type Stride = wrapper.Stride

type DataType = wrapper.DataType

// Numeric is a constraint union collection of valid numeric types for use in tensors
type Numeric interface {
	~int32 | ~int64 | ~float32 | ~float64
}

/*
Tensor essentially acts as a memory-safe wrapper for [wrapper.GraphNode].
Since go does not have a destructor for GraphNode any references to nodes in C memory cannot be cleaned up.
By having a [Close] method, the memory cleanup can be deferred.
*/
type Tensor struct {
	node wrapper.GraphNode
	err  error
}

// String prints the contents of the flint.
// NOTE: If not a isLight flint, this causes the node's graph to be executed!
// Use carefully! (best only in debugging)
func (x Tensor) String() string {
	res, err := wrapper.CalculateResult[float32](x.node)
	if err != nil {
		panic(err)
	}
	return fmt.Sprintf("Tensor (%s): [%v] - type %s", res.Shape, res.Data, res.DataType)
}

/*
init is an internal method to initialize a Tensor struct.
This means setting the reference counter to 1 on wrapper nodes.
This fixes memory management issues, as the node won't be cleared on subsequent calls to [wrapper.OptimizeMemory]

Experimentally we currently use a [finalizer] in this method. This means calling the close method is not required!
However, this may change in the future as we have a massive [finalizer overhead]

[finalizer]: https://medium.com/a-journey-with-go/go-finalizers-786df8e17687
[finalizer overhead]: https://lemire.me/blog/2023/05/19/the-absurd-cost-of-finalizers-in-go/

[finalizer example]: https://gist.github.com/deltamobile/6511901
*/
func (x Tensor) init() {
	if x.err != nil {
		log.Panicf("cannot initialize a errornous flint: %s", x.err)
	}
	wrapper.SetRefCounter(x.node, 1)
	runtime.SetFinalizer(&x, func(x *Tensor) { x.Close() })
	runtime.KeepAlive(x)
}

/*
Close resets the ref counter to 0.
This fixes memory management issues, as the node can be cleared on subsequent calls to [wrapper.OptimizeMemory]
*/
func (x Tensor) Close() {
	fmt.Println("closing a flint ...", x.Shape())
	runtime.SetFinalizer(&x, nil) // remove any finalizer for this flint
	wrapper.SetRefCounter(x.node, 0)
	runtime.KeepAlive(x)
}

func (x Tensor) Shape() Shape {
	return x.node.GetShape()
}

func (x Tensor) Node() wrapper.GraphNode {
	return x.node
}
