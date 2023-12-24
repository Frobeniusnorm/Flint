package flint

import (
	"github.com/Frobeniusnorm/Flint/go/wrapper"
)

// flatten is a recursive function to flatten a slice of any type
func flattenSlice(x any) (result []any, shape Shape, err error) {
	//shape, err = sliceShape(x)
	//if err != nil {
	//	return
	//}
	//
	//for _, item := range x {
	//	switch s := item.(type):
	//		case
	//
	//	result = append(result, s...)
	//}
	return nil, Shape{}, nil
}

func sliceShape(x any) (Shape, error) {
	return Shape{}, nil
}

// isTensor returns whether a given object is a slice or not
func isSlice(x any) bool {
	// TODO: check if reflect.kind matches slice
	return false
}

// isScalar returns whether given object is a scalar (numeric) value or not
func isScalar(x any) bool {
	// TODO: check if kind matches numeric
	//return reflect.
	return false
}

/*
Create a Tensor from a wide range of data types.

- if scalar type just keep it, don't turn into flint yet!
- if nd array flatten it a generate
- if 1d array generate normally
- if [wrapper.GraphNode], just set reference counter

TODO: what if a pointer is passed?
*/
func Create(data any) Tensor {
	if isSlice(data) {
		return Tensor{} // FIXME: CreateFromSlice(data)
	}
	if isScalar(data) {
		return Tensor{} // FIXME: Scalar(data)
	}
	if node, ok := data.(wrapper.GraphNode); ok {
		return FromNode(node)
	}
	panic("invalid data type")
}

//func CreateFromSlice(data any) Tensor {
//	x := flatten(data)
//	shape := Shape{1, 2} // TODO
//
//	flintNode, err := wrapper.CreateGraph(flatten(x), shape)
//	if err != nil {
//		panic(err)
//	}
//	flint.node = &flintNode
//
//	flint.init()
//	return flint
//}

func CreateFromSliceAndShape[T Numeric](data []T, shape Shape) Tensor {
	//data = flatten(data)
	flintNode, err := wrapper.CreateGraph(data, shape)
	tensor := Tensor{node: flintNode, err: err}
	tensor.init()
	return tensor
}

func FromNode(node wrapper.GraphNode) Tensor {
	res := Tensor{node: node}
	res.init()
	return res
}

func FromNodeWithErr(node wrapper.GraphNode, err error) Tensor {
	res := FromNode(node)
	res.err = err
	return res
}

func Scalar[T Numeric](val T) Tensor {
	node, err := wrapper.CreateGraphConstant(val, Shape{1})
	return FromNodeWithErr(node, err)
}

//func Scalar[T Numeric](val T) Tensor {
//	var res Tensor
//	switch any(val).(type) {
//	case int32:
//		var typedVal = any(val).(int32)
//		res.dataInt32 = &typedVal
//		res.dataType = wrapper.INT32
//	case int64:
//		var typedVal = any(val).(int64)
//		res.dataInt64 = &typedVal
//		res.dataType = wrapper.INT64
//	case float32:
//		var typedVal = any(val).(float32)
//		res.dataFloat32 = &typedVal
//		res.dataType = wrapper.FLOAT32
//	case float64:
//		var typedVal = any(val).(float64)
//		res.dataFloat64 = &typedVal
//		res.dataType = wrapper.FLOAT64
//	}
//	res.init()
//	return res
//}

func Constant[T Numeric](val T, shape Shape) Tensor {
	return FromNodeWithErr(wrapper.CreateGraphConstant(val, shape))
}

func Random(shape Shape) Tensor {
	return FromNodeWithErr(wrapper.CreateGraphRandom(shape))
}

//func Arrange(shape Shape, axis int) Tensor {
//	return FromNodeWithErr(wrapper.CreateGraphArrange(shape, axis))
//}
//
//func Identity[T Numeric](size T) Tensor {
//	return FromNodeWithErr(wrapper.CreateGraphIdentity(uint(size)))
//}
//
//// CreateGraphIdentity creates an identity matrix.
//// The datatype will be [INT32]
//func CreateGraphIdentity(size uint) (GraphNode, error) {
//	data := make([]int32, size*size)
//	for i := uint(0); i < size; i++ {
//		data[i+i*size] = int32(1)
//	}
//	return CreateGraph(data, Shape{size, size})
//}
//
//// CreateScalar creates a scalar tensors [(Shape{1})]
//func CreateScalar[T completeNumeric](value T) (GraphNode, error) {
//	return CreateGraph[T]([]T{value}, Shape{1})
//}
//
