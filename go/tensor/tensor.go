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

type DataType = flint.DataType

func greaterDataType(x DataType, y DataType) bool {
	return x > y
}

// Numeric is a constraint union collection of valid numeric types for use in tensors
type Numeric interface {
	~int32 | ~int64 | ~float32 | ~float64
}

/*
Tensor essentially acts as a memory-safe wrapper for [flint.GraphNode].
Since go does not have a destructor for GraphNode any references to nodes in C memory cannot be cleaned up.
By having a [Close] method, the memory cleanup can be deferred.
*/
type Tensor struct {
	node *flint.GraphNode
	err  error
	//data        *any // only in case the actual type doesn't matter. Else use one of the below
	dataInt32   *int32
	dataInt64   *int64
	dataFloat32 *float32
	dataFloat64 *float64
	dataType    DataType
}

// String prints the contents of the tensor.
// NOTE: If not a light tensor, this causes the node's graph to be executed!
// Use carefully! (best only in debugging)
func (x Tensor) String() string {
	if x.light() {
		switch x.dataType {
		case flint.F_INT32:
			return fmt.Sprintf("Tensor (1): [%v] - type %s", *x.dataInt32, x.dataType)
		case flint.F_INT64:
			return fmt.Sprintf("Tensor (1): [%v] - type %s", *x.dataInt64, x.dataType)
		case flint.F_FLOAT32:
			return fmt.Sprintf("Tensor (1): [%v] - type %s", *x.dataFloat32, x.dataType)
		case flint.F_FLOAT64:
			return fmt.Sprintf("Tensor (1): [%v] - type %s", *x.dataFloat64, x.dataType)
		default:
			panic("invalid type")
		}
	} else {
		switch x.dataType {
		case flint.F_INT32:
			res, err := flint.CalculateResult[int32](*x.node)
			if err != nil {
				panic(err)
			}
			return fmt.Sprintf("Tensor (%s): [%v] - typeBefore %s - typeAfter %s", res.Shape, res.Data, x.dataType, res.DataType)

		case flint.F_INT64:
			res, err := flint.CalculateResult[int64](*x.node)
			if err != nil {
				panic(err)
			}
			return fmt.Sprintf("Tensor (%s): [%v] - typeBefore %s - typeAfter %s", res.Shape, res.Data, x.dataType, res.DataType)

		case flint.F_FLOAT32:
			res, err := flint.CalculateResult[float32](*x.node)
			if err != nil {
				panic(err)
			}
			return fmt.Sprintf("Tensor (%s): [%v] - typeBefore %s - typeAfter %s", res.Shape, res.Data, x.dataType, res.DataType)

		case flint.F_FLOAT64:
			res, err := flint.CalculateResult[float64](*x.node)
			if err != nil {
				panic(err)
			}
			return fmt.Sprintf("Tensor (%s): [%v] - typeBefore %s - typeAfter %s", res.Shape, res.Data, x.dataType, res.DataType)
		default:
			panic("invalid datatype?")
		}
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
	if !x.light() {
		flint.SetRefCounter(*x.node, 1)
	}
	runtime.SetFinalizer(&x, func(x *Tensor) { x.Close() }) // FIXME: or only for nodes?
	runtime.KeepAlive(x)
}

/*
Close resets the ref counter to 0.
This fixes memory management issues, as the node can be cleared on subsequent calls to [flint.OptimizeMemory]
*/
func (x Tensor) Close() {
	runtime.SetFinalizer(x, nil) // remove any finalizer for this tensor
	if !x.light() {
		flint.SetRefCounter(*x.node, 0)
	}
	runtime.KeepAlive(x)
}

func (x Tensor) Shape() Shape {
	if x.light() {
		return Shape{1}
	} else {
		return x.node.GetShape()
	}
}

// Value returns the data of a tensor.
// FIXME See String()
//func (x Tensor) Value() (Shape, any) {
//	if x.light() {
//		return Shape{1}, *x.data
//	} else {
//		data, err := flint.CalculateResult[int](*x.node)
//		if err != nil {
//			panic(err)
//		}
//		return data.Shape, data.Data
//	}
//}

// node makes sure that a node is initialized and returns it.
// This function should be called when it is required that a data value (usually from scalar)
// needs to be passed as a node.
// turns a light value into a node
func (x Tensor) readyNode() Tensor {
	if !x.light() {
		return x // no changes needed
	}

	switch x.dataType {
	case flint.F_INT32:
		if x.dataInt32 == nil {
			panic("no data present???")
		}
		flintNode, err := flint.CreateScalar(*x.dataInt32)
		if err != nil {
			panic(err)
		}
		//x.data = nil
		x.node = &flintNode
		x.init()
		return x
	}
	return Tensor{} // TODO: complete
}

// val returns a possible value.
// This is either the node or the value from a light tensor
//func (x Tensor) val() any {
//	if !x.light() {
//		return *x.node
//	} else if x.data != nil {
//		return *x.data
//	} else {
//		panic("no value present - invalid tensor?")
//	}
//}

func (x Tensor) light() bool {
	return x.node == nil
}

func lightVal[T Numeric](x Tensor) T {
	// TODO: assert is x is light!
	switch x.dataType {
	case flint.F_INT32:
		return T(*x.dataInt32)
	case flint.F_INT64:
		return T(*x.dataInt64)
	case flint.F_FLOAT32:
		return T(*x.dataFloat32)
	case flint.F_FLOAT64:
		return T(*x.dataFloat64)
	default:
		panic("invalid data type")
	}
}

// To converts the tensor's data type
// FIXME: set err instead of panic?
func (x Tensor) To(tt DataType) Tensor {
	if x.light() {
		switch tt {
		case flint.F_INT32:
			return Scalar(lightVal[int32](x))
		case flint.F_INT64:
			return Scalar(lightVal[int64](x))
		case flint.F_FLOAT32:
			return Scalar(lightVal[float32](x))
		case flint.F_FLOAT64:
			return Scalar(lightVal[float64](x))
		default:
			panic("invalid type")
		}
	} else {
		flintNode, err := flint.Convert(*x.node, tt)
		return FromNodeWithErr(flintNode, err)
	}
}
