package flint

import (
	"github.com/Frobeniusnorm/Flint/go/wrapper"
)

func CreateFromSliceAndShape(data any, shape Shape, dataType DataType) Tensor {
	switch dataType {
	case wrapper.INT32:
		x := flatten[Int](data)
		return FromNodeWithErr(wrapper.CreateGraphDataInt(x, shape))
	case wrapper.INT64:
		x := flatten[Long](data)
		return FromNodeWithErr(wrapper.CreateGraphDataLong(x, shape))
	case wrapper.FLOAT32:
		x := flatten[Float](data)
		return FromNodeWithErr(wrapper.CreateGraphDataFloat(x, shape))
	case wrapper.FLOAT64:
		x := flatten[Double](data)
		return FromNodeWithErr(wrapper.CreateGraphDataDouble(x, shape))
	default:
		panic("invalid type")
	}
}

func CreateFromSlice(data any, dataType DataType) Tensor {
	switch dataType {
	case wrapper.INT32:
		x := flatten[Int](data)
		return FromNodeWithErr(wrapper.CreateGraphDataInt(x, Shape{uint(len(x))}))
	case wrapper.INT64:
		x := flatten[Long](data)
		return FromNodeWithErr(wrapper.CreateGraphDataLong(x, Shape{uint(len(x))}))
	case wrapper.FLOAT32:
		x := flatten[Float](data)
		return FromNodeWithErr(wrapper.CreateGraphDataFloat(x, Shape{uint(len(x))}))
	case wrapper.FLOAT64:
		x := flatten[Double](data)
		return FromNodeWithErr(wrapper.CreateGraphDataDouble(x, Shape{uint(len(x))}))
	default:
		panic("invalid type")
	}
}

func Scalar[T Numeric](val T) Tensor {
	switch any(val).(type) {
	case Int:
		return FromNodeWithErr(wrapper.CreateGraphConstantInt(Int(val), Shape{1}))
	case Long:
		return FromNodeWithErr(wrapper.CreateGraphConstantLong(Long(val), Shape{1}))
	case Float:
		return FromNodeWithErr(wrapper.CreateGraphConstantFloat(Float(val), Shape{1}))
	case Double:
		return FromNodeWithErr(wrapper.CreateGraphConstantDouble(Double(val), Shape{1}))
	default:
		panic("invalid type")
	}
}

func Constant[T Numeric](val T, shape Shape) Tensor {
	switch any(val).(type) {
	case Int:
		return FromNodeWithErr(wrapper.CreateGraphConstantInt(Int(val), shape))
	case Long:
		return FromNodeWithErr(wrapper.CreateGraphConstantLong(Long(val), shape))
	case Float:
		return FromNodeWithErr(wrapper.CreateGraphConstantFloat(Float(val), shape))
	case Double:
		return FromNodeWithErr(wrapper.CreateGraphConstantDouble(Double(val), shape))
	default:
		panic("invalid type")
	}
}

func Random(shape Shape) Tensor {
	return FromNodeWithErr(wrapper.CreateGraphRandom(shape))
}

func Arrange(shape Shape, axis int) Tensor {
	return FromNodeWithErr(wrapper.CreateGraphArrange(shape, axis))
}

// Identity creates an identity matrix.
// The datatype will be [INT32]
func Identity(size uint) Tensor {
	data := make([]int32, size*size)
	for i := uint(0); i < size; i++ {
		data[i+i*size] = int32(1)
	}
	return FromNodeWithErr(wrapper.CreateGraphDataInt(data, Shape{size, size}))
}

//func OneHot(node GraphNode, numClasses uint) (GraphNode, error) {
//	id, err := CreateGraphIdentity(numClasses)
//	if err != nil {
//		return GraphNode{}, err
//	}
//
//	id, err = Index(id, node)
//	if err != nil {
//		return GraphNode{}, err
//	}
//
//	return id, nil
//}
