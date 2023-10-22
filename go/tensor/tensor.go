/*
Package tensor serves as a wrapper around the specific tensor library.
It abstracts away all technicalities like memory management and error handling.

This package should be used to handle all tensor operations instead of relying on the underlying Library alone and risking memory issues.
*/
package tensor

import "github.com/Frobeniusnorm/Flint/go/flint"

/*
Tensor essentially acts as a memory-safe wrapper for [flint.GraphNode].
Since go does not have a destructor for GraphNode any references to nodes in C memory cannot be cleaned up.
By having a [Close] method, the memory cleanup can be deferred.
*/
type Tensor struct {
	Node flint.GraphNode
}

/*
NewTensor returns a properly initialized [Tensor] from a [flint.GraphNode]
*/
func NewTensor(node flint.GraphNode) Tensor {
	tensor := Tensor{Node: node}
	tensor.New()
	return tensor
}

/*
New sets the ref counter to 1.
This fixes memory management issues, as the node won't be cleared on subsequent calls to [flint.OptimizeMemory]
*/
func (x Tensor) New() {
	flint.SetRefCounter(x.Node, 1)
}

/*
Close resets the ref counter to 0.
This fixes memory management issues, as the node can be cleared on subsequent calls to [flint.OptimizeMemory]
*/
func (x Tensor) Close() {
	flint.SetRefCounter(x.Node, 0)
}

func (x Tensor) Shape() Shape {
	return Shape(x.Node.GetShape())
}

// Value returns the data of a tensor.
func (x Tensor) Value() any {
	return nil
}
