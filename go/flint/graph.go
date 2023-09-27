package flint

// #include <flint/flint.h>
import "C"
import "unsafe"

/*
CreateGraph creates a Graph with a single store instruction, the data is copied to internal memory.

Params:
  - [data]: the flattened data array that should be loaded into the [GraphNode]
  - Type params [T]: the datatype of [data]
  - [shape]: Each entry describing the size of the corresponding dimension.
  - [datatype]: Specifying a valid flint [DataType]
*/
func CreateGraph[T completeNumbers](data []T, shape Shape, datatype DataType) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)
	newData := convertArray[T, C.float](data)

	var flintNode *C.FGraphNode = C.fCreateGraph(unsafe.Pointer(&(newData[0])),
		C.int(len(data)), uint32(datatype), &(newShape[0]), C.int(len(shape)))
	return GraphNode{ref: flintNode}
}

/*
CreateGraphConstant creates a tensor in a specified [DataType] that contains the single given values in all entries.

Params:
  - [value]: the value this tensor should consist of
  - [shape]: Each entry describing the size of the corresponding dimension.
  - [datatype]: Specifying a valid flint [DataType]
*/
func CreateGraphConstant[T numeric](value T, shape Shape, datatype DataType) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)
	dimensions := C.int(len(shape))

	var flintNode *C.FGraphNode
	switch datatype {
	case F_INT32:
		flintNode = C.fconstant_i(C.int(value), &(newShape[0]), dimensions)
	case F_INT64:
		flintNode = C.fconstant_l(C.long(value), &(newShape[0]), dimensions)
	case F_FLOAT32:
		flintNode = C.fconstant_f(C.float(value), &(newShape[0]), dimensions)
	case F_FLOAT64:
		flintNode = C.fconstant_d(C.double(value), &(newShape[0]), dimensions)
	default:
		panic("invalid data type")
	}
	return GraphNode{ref: flintNode}
}

/*
CreateGraphRandom creates a [F_FLOAT32] tensor that contains randomly distributed values in the range of [0, 1)

Params:
  - [shape]: Each entry describing the size of the corresponding dimension.
*/
func CreateGraphRandom(shape Shape) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)

	var flintNode *C.FGraphNode = C.frandom(&(newShape[0]), C.int(len(shape)))
	return GraphNode{ref: flintNode}
}

/*
CreateGraphArrange creates a [F_INT64] tensor that contains the indices relative to a given dimension [axis] for each element.
i.e. each entry is its index in that corresponding dimension.
If you need to index more than one dimension, create multiple such tensors with [CreateGraphArrange].
*/
func CreateGraphArrange(shape Shape, axis int) GraphNode {
	newShape := convertArray[uint, C.size_t](shape)

	var flintNode *C.FGraphNode = C.farange(&(newShape[0]), C.int(len(shape)), C.int(axis))
	return GraphNode{ref: flintNode}
}
