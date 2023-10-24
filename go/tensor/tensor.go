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
	"runtime"
)

type Shape = flint.Shape // this way we keep the functions etc.

type Axes = flint.Axes

// Numeric is a constraint union collection of valid numeric types for use in tensors
type Numeric interface {
	~int32 | ~int64 | ~float32 | ~float64
}

/*
Tensor essentially acts as a memory-safe wrapper for [flint.GraphNode].
Since go does not have a destructor for GraphNode any references to nodes in C memory cannot be cleaned up.
By having a [Close] method, the memory cleanup can be deferred.
*/
type Tensor[T Numeric] struct {
	node *flint.GraphNode
	err  error
	data *T // only in case the actual type doesn't matter. Else use one of the below
}

// String prints the contents of the tensor.
// NOTE: If not a light tensor, this causes the node's graph to be executed!
// Use carefully! (best only in debugging)
func (x Tensor[T]) String() string {
	if x.light() {
		return fmt.Sprintf("Tensor (1): [%v]", *x.data)
	} else {
		res, err := flint.CalculateResult[T](*x.node)
		if err != nil {
			panic(err)
		}
		return fmt.Sprintf("Tensor (%s): [%v]", res.Shape, res.Data)
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
func (x Tensor[T]) init() {
	if !x.light() {
		flint.SetRefCounter(*x.node, 1)
	}
	runtime.SetFinalizer(&x, func(x *Tensor[T]) { x.Close() }) // FIXME: or only for nodes?
	runtime.KeepAlive(x)
}

/*
Close resets the ref counter to 0.
This fixes memory management issues, as the node can be cleared on subsequent calls to [flint.OptimizeMemory]
*/
func (x Tensor[T]) Close() {
	runtime.SetFinalizer(x, nil) // remove any finalizer for this tensor
	if !x.light() {
		flint.SetRefCounter(*x.node, 0)
	}
	runtime.KeepAlive(x)
}

func (x Tensor[T]) Shape() Shape {
	if x.light() {
		return Shape{1}
	} else {
		return x.node.GetShape()
	}
}

// Value returns the data of a tensor.
// FIXME
func (x Tensor[T]) Value() (Shape, any) {
	if x.light() {
		return Shape{1}, *x.data
	} else {
		data, err := flint.CalculateResult[int](*x.node)
		if err != nil {
			panic(err)
		}
		return data.Shape, data.Data
	}
}

// node makes sure that a node is initialized and returns it.
// This function should be called when it is required that a data value (usually from scalar)
// needs to be passed as a node.
func (x Tensor[T]) readyNode() Tensor[T] {
	if !x.light() {
		return x // no changes needed
	}

	flintNode, err := flint.CreateScalar(*x.data)
	if err != nil {
		panic(err)
	}
	x.data = nil
	x.node = &flintNode
	x.init()
	return x
}

// val returns a possible value.
// This is either the node or the value from a light tensor
func (x Tensor[T]) val() any {
	if !x.light() {
		return *x.node
	} else if x.data != nil {
		return *x.data
	} else {
		panic("no value present - invalid tensor?")
	}
}

func (x Tensor[T]) light() bool {
	return x.node == nil
}

func lightLast[T Numeric](x Tensor[T], y Tensor[T]) (a Tensor[T], b Tensor[T]) {
	if !x.light() {
		return x, y
	} else if !y.light() {
		return y, x
	} else {
		return x.readyNode(), y
	}
}

func (x Tensor[T]) ToFloat32() Tensor[float32] {
	var res Tensor[float32]
	if x.light() {
		val := float32(*x.data)
		res.data = &val
	} else {
		flintNode, err := flint.Convert(*x.node, flint.F_FLOAT32)
		if err != nil {
			panic(err)
		}
		res.node = &flintNode
	}
	res.init()
	return res
}

func (x Tensor[T]) ToFloat64() Tensor[float64] {
	var res Tensor[float64]
	if x.light() {
		val := float64(*x.data)
		res.data = &val
	} else {
		// FIXME: why is x.node nil?
		flintNode, err := flint.Convert(*x.node, flint.F_FLOAT64)
		if err != nil {
			panic(err)
		}
		res.node = &flintNode
	}
	res.init()
	return res
}

func (x Tensor[T]) ToInt32() Tensor[int32] {
	var res Tensor[int32]
	if x.light() {
		val := int32(*x.data)
		res.data = &val
	} else {
		flintNode, err := flint.Convert(*x.node, flint.F_INT32)
		if err != nil {
			panic(err)
		}
		res.node = &flintNode
	}
	res.init()
	return res
}

func (x Tensor[T]) ToInt64() Tensor[int64] {
	var res Tensor[int64]
	if x.light() {
		val := int64(*x.data)
		res.data = &val
	} else {
		flintNode, err := flint.Convert(*x.node, flint.F_INT64)
		if err != nil {
			panic(err)
		}
		res.node = &flintNode
	}
	res.init()
	return res
}
