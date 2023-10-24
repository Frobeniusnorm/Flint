package tensor

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
