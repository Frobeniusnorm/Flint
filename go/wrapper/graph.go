package wrapper

// #include <flint/flint.h>
import "C"
import "unsafe"

/*
CreateGraphDataInt creates a [INT32] flint that contains the given data.

The backend creates a graph with a single store instruction.
The data is copied to internal memory, so after return of the function,
`data` and `shape` may be deleted
*/
func CreateGraphDataInt(data []Int, shape Shape) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)
	newData := convertArray[Int, C.int](data)
	flintNode, errno := C.fCreateGraph(unsafe.Pointer(&(newData[0])), C.int(len(data)), C.enum_FType(INT32), &(newShape[0]), C.int(len(shape)))
	return returnHelper(flintNode, errno)
}

/*
CreateGraphDataLong creates a [INT64] flint that contains the given data.

The backend creates a graph with a single store instruction.
The data is copied to internal memory, so after return of the function,
`data` and `shape` may be deleted
*/
func CreateGraphDataLong(data []Long, shape Shape) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)
	newData := convertArray[Long, C.long](data)
	flintNode, errno := C.fCreateGraph(unsafe.Pointer(&(newData[0])), C.int(len(data)), C.enum_FType(INT64), &(newShape[0]), C.int(len(shape)))
	return returnHelper(flintNode, errno)
}

/*
CreateGraphDataFloat creates a [FLOAT32] flint that contains the given data.

The backend creates a graph with a single store instruction.
The data is copied to internal memory, so after return of the function,
`data` and `shape` may be deleted
*/
func CreateGraphDataFloat(data []Float, shape Shape) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)
	newData := convertArray[Float, C.float](data)
	flintNode, errno := C.fCreateGraph(unsafe.Pointer(&(newData[0])), C.int(len(data)), C.enum_FType(FLOAT32), &(newShape[0]), C.int(len(shape)))
	return returnHelper(flintNode, errno)
}

/*
CreateGraphDataDouble creates a [FLOAT64] flint that contains the given data.

The backend creates a graph with a single store instruction.
The data is copied to internal memory, so after return of the function,
`data` and `shape` may be deleted
*/
func CreateGraphDataDouble(data []Double, shape Shape) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)
	newData := convertArray[Double, C.double](data)
	flintNode, errno := C.fCreateGraph(unsafe.Pointer(&(newData[0])), C.int(len(data)), C.enum_FType(FLOAT64), &(newShape[0]), C.int(len(shape)))
	return returnHelper(flintNode, errno)
}

func CreateGraphConstantInt(value Int, shape Shape) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)
	flintNode, errno := C.fconstant_i(C.int(value), &(newShape[0]), C.int(len(shape)))
	return returnHelper(flintNode, errno)
}

func CreateGraphConstantLong(value Long, shape Shape) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)
	flintNode, errno := C.fconstant_l(C.long(value), &(newShape[0]), C.int(len(shape)))
	return returnHelper(flintNode, errno)
}

func CreateGraphConstantFloat(value Float, shape Shape) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)
	flintNode, errno := C.fconstant_f(C.float(value), &(newShape[0]), C.int(len(shape)))
	return returnHelper(flintNode, errno)
}

func CreateGraphConstantDouble(value Double, shape Shape) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)
	flintNode, errno := C.fconstant_d(C.double(value), &(newShape[0]), C.int(len(shape)))
	return returnHelper(flintNode, errno)
}

/*
CreateGraphRandom creates a [FLOAT64] flint that contains randomly distributed values in the range of [0, 1)

Params:
  - [shape]: Each entry describing the size of the corresponding dimension.
*/
func CreateGraphRandom(shape Shape) (GraphNode, error) {
	newShape := convertArray[uint, C.size_t](shape)
	flintNode, errno := C.frandom(&(newShape[0]), C.int(len(shape)))
	return returnHelper(flintNode, errno)
}

/*
CreateGraphArrange creates a [INT64] flint that contains the indices relative to a given dimension [axis] for each element.
i.e. each entry is its index in that corresponding dimension.
If you need to index more than one dimension, create multiple such tensors with [CreateGraphArrange].
*/
func CreateGraphArrange(shape Shape, axis int) (GraphNode, error) {
	validAxis := axis
	if validAxis < 0 || validAxis >= len(shape) {
		return GraphNode{}, buildErrorCustom(ErrIllegalDimensionality, "invalid axis")
	}

	newShape := convertArray[uint, C.size_t](shape)
	flintNode, errno := C.farange(&(newShape[0]), C.int(len(shape)), C.int(axis))
	return returnHelper(flintNode, errno)
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
