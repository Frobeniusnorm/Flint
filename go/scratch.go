package scratch

import "fmt"

//func convertToCNumber[In completeNumeric, Out cNumbers](n In) Out {
//	switch v := any(n).(type) {
//	case int, int8, int16, int32:
//		return C.int(v)
//	case int64:
//		return C.long(v)
//	case uint, uint8, uint16, uint32:
//		return C.uint(v)
//	case float32:
//		return C.float(v)
//	case float64:
//		return C.double(v)
//	default:
//		// Handle this case appropriately; maybe panic or return a zero value
//		panic("unsupported type")
//	}
//}

// use for type debugging
func describe(i any) {
	fmt.Printf("describe (value, underlying type): (%v, %T)\n", i, i)
}

// See: https://stackoverflow.com/questions/71587996/cannot-use-type-assertion-on-type-parameter-value
//func convertArrayOld[In completeNumeric, Out cNumbers](arr []In) []Out {
//	result := make([]Out, len(arr))
//	for idx, val := range arr {
//		result[idx] = Out(val) //nolint:all
//		//x := any(val).(Out)
//		//result[idx] = x
//	}
//	return result
//}

// Max returns the maximum value across all dimensions
// see: https://numpy.org/doc/stable/reference/generated/numpy.max.html
// FIXME: func Max(a GraphNode, axes Axis, scalar bool, keepDims bool) {
func Max(node GraphNode) (GraphNode, error) {
	//for !node.GetShape().Equal(Shape{1}) { }
	node, err := Flatten(node)
	if err != nil {
		return GraphNode{}, err
	}

	node, err = ReduceMax(node, 0)
	if err != nil {
		return GraphNode{}, err
	}

	return node, nil
}

// Min returns the minimum value across all dimensions
func Min(node GraphNode) (GraphNode, error) {
	node, err := Flatten(node)
	if err != nil {
		return GraphNode{}, err
	}

	node, err = ReduceMin(node, 0)
	if err != nil {
		return GraphNode{}, err
	}

	return node, nil
}

// Sum sums up all values in the flattened flint
func Sum(node GraphNode) (GraphNode, error) {
	node, err := Flatten(node)
	if err != nil {
		return GraphNode{}, err
	}

	node, err = ReduceSum(node, 0)
	if err != nil {
		return GraphNode{}, err
	}

	return node, nil
}

// TransposeFull acts like a traditional transpose where the order of all axis are reversed
func TransposeFull(node GraphNode) (GraphNode, error) {
	n := len(node.GetShape())
	ax := make(Axes, n)
	for i := 0; i < n; i++ {
		ax[i] = uint(n - i - 1)
	}
	return Transpose(node, ax)
}

func OneHot(node GraphNode, numClasses uint) (GraphNode, error) {
	id, err := CreateGraphIdentity(numClasses)
	if err != nil {
		return GraphNode{}, err
	}

	id, err = Index(id, node)
	if err != nil {
		return GraphNode{}, err
	}

	return id, nil
}

func (x Tensor) Max() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Max(*x.node))
}

func (x Tensor) Min() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Min(*x.node))
}

func (x Tensor) Sum() Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.Sum(*x.node))
}

func (x Tensor) OneHot(numClasses uint) Tensor {
	x = x.readyNode()
	return FromNodeWithErr(wrapper.OneHot(*x.node, numClasses))
}

func largerType(x Tensor, y Tensor) DataType {
	if x.dataType > y.dataType {
		return x.dataType
	} else {
		return y.dataType
	}
}

func lightLast(x Tensor, y Tensor) (a Tensor, b Tensor) {
	if !x.isLight() {
		return x, y
	} else if !y.isLight() {
		return y, x
	} else {
		return x.readyNode(), y
	}
}
