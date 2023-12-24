package flint

// CreateGraphIdentity creates an identity matrix.
// The datatype will be [F_INT32]
func CreateGraphIdentity(size uint) (GraphNode, error) {
	data := make([]int32, size*size)
	for i := uint(0); i < size; i++ {
		data[i+i*size] = int32(1)
	}
	return CreateGraph(data, Shape{size, size})
}

// CreateScalar creates a scalar tensors [(Shape{1})]
func CreateScalar[T completeNumeric](value T) (GraphNode, error) {
	return CreateGraph[T]([]T{value}, Shape{1})
}
