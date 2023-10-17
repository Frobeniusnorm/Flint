package flint

import "fmt"

func ExampleReduceMax() {
	node1 := CreateGraph([]int{1, 32, 3, 4, 5, 3}, Shape{2, 3})
	node2 := CreateGraph([]int{9, 2, 3, -1, 5, 6}, Shape{2, 3})

	node1 = ReduceMax(node1, 0)
	node2 = ReduceMax(node2, 1)

	res1 := CalculateResult[int](node1)
	res2 := CalculateResult[int](node2)

	fmt.Printf("Data: %s, Shape: %s\n", res1.Data, res1.Shape)
	fmt.Printf("Data: %s, Shape: %s\n", res2.Data, res2.Shape)

	// Output:
	// Data: [4 32 3], Shape: [3]
	// Data: [9 6], Shape: [2]
}

func ExampleExpand() {
	node1 := CreateGraph([]int{1, 2}, Shape{2})
	node2 := node1

	node1 = Expand(node1, uint(1), 3)
	node2 = Expand(node2, uint(0), 2)

	res1 := CalculateResult[int](node1)
	res2 := CalculateResult[int](node2)

	fmt.Printf("Data: %s, Shape: %s\n", res1.Data, res1.Shape)
	fmt.Printf("Data: %s, Shape: %s\n", res2.Data, res2.Shape)

	// Output:
	// Data: [1 1 1 2 2 2], Shape: [2 3]
	// Data: [1 2 1 ], Shape: [2 2]
}

func ExampleMax() {
	node := CreateGraph([]int{1, 32, 3, 4, 5, 3}, Shape{6})
	node = Max(node)
	res := CalculateResult[int](node)
	fmt.Println(res.Data[0])
	// Output: 32
}

func ExampleMin() {
	node := CreateGraph([]int{1, 32, -3, 4, 5, 3}, Shape{6})
	node = Min(node)
	res := CalculateResult[int](node)
	fmt.Println(res.Data[0])
	// Output: -3
}

func ExampleOneHot() {
	node := CreateGraph([]int{0, 1, 2, 1, 2, 0}, Shape{6})
	node = OneHot(node, 3)
	res := CalculateResult[int](node)
	fmt.Printf("Data: %s, Shape: %s\n", res.Data, res.Shape)
	// Output:
	// Data: [1 0 0 0 1 0 0 0 1 0 1 0 0 0 1 1 0 0], Shape: [6 3]
}
