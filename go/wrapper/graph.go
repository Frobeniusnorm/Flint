package wrapper

// #include <flint/flint.h>
import "C"
import (
	"unsafe"
)

/*
CreateGraph creates a Graph with a single store instruction, the data is copied to internal memory.

Params:
  - [data]: the flattened data array that should be loaded into the [GraphNode]
  - Type params [T]: the datatype of [data]
  - [shape]: Each entry describing the size of the corresponding dimension.
  - [datatype]: Specifying a valid wrapper [dataType]
*/
func CreateGraph[T completeNumeric](data []T, shape Shape) (GraphNode, error) {
	//if len(data) != int(shape.NumItems()) {
	//	return GraphNode{}, Error{
	//		Message: fmt.Sprintf("data (len: %d) and shape (numItems: %d) do not match", len(data), int(shape.NumItems())),
	//		Err:     ErrIncompatibleShapes,
	//	}
	//}
	datatype := closestType(data[0])

	// The C function does not perform casts. The data for the void pointer must match the FType!
	newShape := convertArray[uint, C.size_t](shape)
	var newData unsafe.Pointer

	switch datatype {
	case INT32:
		convertedData := convertArray[T, C.int](data)
		newData = unsafe.Pointer(&(convertedData[0]))
	case INT64:
		convertedData := convertArray[T, C.long](data)
		newData = unsafe.Pointer(&(convertedData[0]))
	case FLOAT32:
		convertedData := convertArray[T, C.float](data)
		newData = unsafe.Pointer(&(convertedData[0]))
	case FLOAT64:
		convertedData := convertArray[T, C.double](data)
		newData = unsafe.Pointer(&(convertedData[0]))
	default:
		panic("invalid type")
	}

	flintNode, errno := C.fCreateGraph(newData, C.int(len(data)), C.enum_FType(datatype), &(newShape[0]), C.int(len(shape)))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
CreateGraphConstant creates a flint in a specified [dataType] that contains the single given values in all entries.

Params:
  - [value]: the value this flint should consist of
  - [shape]: Each entry describing the size of the corresponding dimension.
  - [datatype]: Specifying a valid wrapper [dataType]
*/
func CreateGraphConstant[T completeNumeric](value T, shape Shape) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)
	datatype := closestType(value)
	dimensions := C.int(len(shape))

	var flintNode *C.FGraphNode
	var errno error
	switch datatype {
	case INT32:
		flintNode, errno = C.fconstant_i(C.int(value), &(newShape[0]), dimensions)
	case INT64:
		flintNode, errno = C.fconstant_l(C.long(value), &(newShape[0]), dimensions)
	case FLOAT32:
		flintNode, errno = C.fconstant_f(C.float(value), &(newShape[0]), dimensions)
	case FLOAT64:
		flintNode, errno = C.fconstant_d(C.double(value), &(newShape[0]), dimensions)
	default:
		panic("invalid type")
	}
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
CreateGraphRandom creates a [FLOAT64] flint that contains randomly distributed values in the range of [0, 1)

Params:
  - [shape]: Each entry describing the size of the corresponding dimension.
*/
func CreateGraphRandom(shape Shape) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)

	flintNode, errno := C.frandom(&(newShape[0]), C.int(len(shape)))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

/*
CreateGraphArrange creates a [INT64] flint that contains the indices relative to a given dimension [axis] for each element.
i.e. each entry is its index in that corresponding dimension.
If you need to index more than one dimension, create multiple such tensors with [CreateGraphArrange].
*/
func CreateGraphArrange(shape Shape, axis int) (GraphNode, error) {
	validAxis := axis
	if validAxis < 0 || validAxis >= len(shape) {
		return GraphNode{}, Error{
			Message: "invalid axis",
			Err:     ErrIllegalDimensionality,
		}
	}

	newShape := convertArray[uint, C.size_t](shape)

	flintNode, errno := C.farange(&(newShape[0]), C.int(len(shape)), C.int(axis))
	if flintNode == nil {
		return GraphNode{}, buildError(errno)
	}
	return GraphNode{ref: flintNode}, nil
}

// GetShape returns the shape of a graph node
// NOTE: this will not execute the graph or change anything in memory
// TODO: remove??
func (node GraphNode) GetShape() Shape {
	var flintNode *C.FGraphNode = node.ref
	shapePtr := unsafe.Pointer(flintNode.operation.shape)
	shapeSize := int(flintNode.operation.dimensions)
	return fromCToArray[uint](shapePtr, shapeSize, INT64)
}

func (node GraphNode) GetType() DataType {
	var flintNode *C.FGraphNode = node.ref
	return DataType(flintNode.operation.data_type)
}