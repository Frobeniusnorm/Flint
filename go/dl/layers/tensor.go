package layers

import "github.com/Frobeniusnorm/Flint/go/flint"

/*
Tensor essentially acts as a memory-safe wrapper for [flint.GraphNode].
Since go does not have a destructor for GraphNode any references to nodes in C memory cannot be cleaned up.
By having a [Close] method that satisfies [io.Closer] and can be deferred closed this issue is resolved.
*/
type Tensor struct {
	node flint.GraphNode
}

/*
NewTensor returns a properly initialized [Tensor] from a [flint.GraphNode]
*/
func NewTensor(node flint.GraphNode) Tensor {
	tensor := Tensor{node: node}
	tensor.New()
	return tensor
}

/*
New sets the ref counter to 1.
This fixes memory management issues, as the node won't be cleared on subsequent calls to [flint.OptimizeMemory]
*/
func (t *Tensor) New() {
	flint.SetRefCounter(t.node, 1)
}

/*
Close resets the ref counter to 0.
This fixes memory management issues, as the node can be cleared on subsequent calls to [flint.OptimizeMemory]
*/
func (t *Tensor) Close() {
	flint.SetRefCounter(t.node, 0)
}
