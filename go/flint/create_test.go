package flint

import (
	"fmt"
	"testing"
)

//func TestCreateFromSlice(t *testing.T) {
//	x := [][]uint{
//		{1, 2, 5, 5},
//		{1, 2, 3, 3},
//	}
//	t := CreateFromSlice(x)
//	fmt.Println(t)
//}

func TestFlattenSlice(t *testing.T) {
	x := [][][]int{
		{{1, 2, 3}, {1, 2}},
		{{2, 3, 4}, {4, 5, 56}},
	}
	fmt.Println(len(x))
	FlattenSlice(x)
}

func TestInferShape(t *testing.T) {
	x := [][]int{
		{1, 2},
		{2, 3},
	}
	fmt.Println(len(x))
	InferShape(x)
}

//func ExampleMax() {
//	node, _ := CreateGraph([]int{1, 32, 3, 4, 5, 3}, Shape{6})
//	node, _ = Max(node)
//	res, _ := CalculateResult[int](node)
//	fmt.Println(res.Data[0])
//	// Output: 32
//}
//
//func ExampleMin() {
//	node, _ := CreateGraph([]int{1, 32, -3, 4, 5, 3}, Shape{6})
//	node, _ = Min(node)
//	res, _ := CalculateResult[int](node)
//	fmt.Println(res.Data[0])
//	// Output: -3
//}

//func ExampleOneHot() {
//	node, _ := CreateGraph([]int{0, 1, 2, 1, 2, 0}, Shape{6})
//	node, _ = OneHot(node, 3)
//	res, _ := CalculateResult[int](node)
//	fmt.Printf("Data: %s, Shape: %s\n", res.Data, res.Shape)
//	// Output:
//	// Data: [1 0 0 0 1 0 0 0 1 0 1 0 0 0 1 1 0 0], Shape: [6 3]
//}
