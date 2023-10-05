package flint

// #include <flint/flint.h>
import "C"
import (
	"log"
	"unsafe"
)

/*
CreateGraph creates a Graph with a single store instruction, the data is copied to internal memory.

Params:
  - [data]: the flattened data array that should be loaded into the [GraphNode]
  - Type params [T]: the datatype of [data]
  - [shape]: Each entry describing the size of the corresponding dimension.
  - [datatype]: Specifying a valid flint [DataType]
*/
func CreateGraph[T completeNumeric](data []T, shape Shape) GraphNode {
	if len(data) != int(shape.NumItems()) {
		log.Panicf("data (len: %d) and shape (numItems: %d) do not match", len(data), int(shape.NumItems()))
	}
	datatype := closestType(data[0])

	// The C function does not perform casts. The data for the void pointer must match the FType!

	newShape := convertArray[uint, C.size_t](shape)
	var newData unsafe.Pointer
	switch datatype {
	case f_INT32:
		convertedData := convertArray[T, C.int](data)
		newData = unsafe.Pointer(&(convertedData[0]))
	case f_INT64:
		convertedData := convertArray[T, C.long](data)
		newData = unsafe.Pointer(&(convertedData[0]))
	case f_FLOAT32:
		convertedData := convertArray[T, C.float](data)
		newData = unsafe.Pointer(&(convertedData[0]))
	case f_FLOAT64:
		convertedData := convertArray[T, C.double](data)
		newData = unsafe.Pointer(&(convertedData[0]))
	default:
		panic("invalid data type")
	}

	var flintNode *C.FGraphNode = C.fCreateGraph(newData, C.int(len(data)), C.enum_FType(datatype), &(newShape[0]), C.int(len(shape)))
	return GraphNode{ref: flintNode}
}

// CreateScalar creates a scalar tensors [(Shape{1})]
func CreateScalar[T completeNumeric](value T) GraphNode {
	return CreateGraph[T]([]T{value}, Shape{1})
}

/*
CreateGraphConstant creates a tensor in a specified [DataType] that contains the single given values in all entries.

Params:
  - [value]: the value this tensor should consist of
  - [shape]: Each entry describing the size of the corresponding dimension.
  - [datatype]: Specifying a valid flint [DataType]
*/
func CreateGraphConstant[T completeNumeric](value T, shape Shape) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)
	datatype := closestType(value)
	dimensions := C.int(len(shape))

	var flintNode *C.FGraphNode
	switch datatype {
	case f_INT32:
		flintNode = C.fconstant_i(C.int(value), &(newShape[0]), dimensions)
	case f_INT64:
		flintNode = C.fconstant_l(C.long(value), &(newShape[0]), dimensions)
	case f_FLOAT32:
		flintNode = C.fconstant_f(C.float(value), &(newShape[0]), dimensions)
	case f_FLOAT64:
		flintNode = C.fconstant_d(C.double(value), &(newShape[0]), dimensions)
	default:
		panic("invalid data type")
	}
	return GraphNode{ref: flintNode}
}

/*
CreateGraphRandom creates a [f_FLOAT64] tensor that contains randomly distributed values in the range of [0, 1)

Params:
  - [shape]: Each entry describing the size of the corresponding dimension.
*/
func CreateGraphRandom(shape Shape) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)

	var flintNode *C.FGraphNode = C.frandom(&(newShape[0]), C.int(len(shape)))
	return GraphNode{ref: flintNode}
}

/*
CreateGraphArrange creates a [f_INT64] tensor that contains the indices relative to a given dimension [axis] for each element.
i.e. each entry is its index in that corresponding dimension.
If you need to index more than one dimension, create multiple such tensors with [CreateGraphArrange].
*/
func CreateGraphArrange(shape Shape, axis int) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)

	var flintNode *C.FGraphNode = C.farange(&(newShape[0]), C.int(len(shape)), C.int(axis))
	return GraphNode{ref: flintNode}
}

// GetShape returns the shape of a graph node
// NOTE: this will not execute the graph or change anything in memory
func (node GraphNode) GetShape() Shape {
	var flintNode *C.FGraphNode = node.ref
	shapePtr := unsafe.Pointer(flintNode.operation.shape)
	shapeSize := int(flintNode.operation.dimensions)
	return fromCToArray[uint](shapePtr, shapeSize, f_INT64)
}
