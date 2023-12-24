package wrapper

import "fmt"

func ExampleReduceMax() {
	node1, _ := CreateGraph([]int{1, 32, 3, 4, 5, 3}, Shape{2, 3})
	node2, _ := CreateGraph([]int{9, 2, 3, -1, 5, 6}, Shape{2, 3})

	node1, _ = ReduceMax(node1, 0)
	node2, _ = ReduceMax(node2, 1)

	res1, _ := CalculateResult[int](node1)
	res2, _ := CalculateResult[int](node2)

	fmt.Printf("Data: %s, Shape: %s\n", res1.Data, res1.Shape)
	fmt.Printf("Data: %s, Shape: %s\n", res2.Data, res2.Shape)

	// Output:
	// Data: [4 32 3], Shape: [3]
	// Data: [9 6], Shape: [2]
}

func ExampleExpand() {
	node1, _ := CreateGraph([]int{1, 2}, Shape{2})
	node2 := node1

	node1, _ = Expand(node1, uint(1), 3)
	node2, _ = Expand(node2, uint(0), 2)

	res1, _ := CalculateResult[int](node1)
	res2, _ := CalculateResult[int](node2)

	fmt.Printf("Data: %s, Shape: %s\n", res1.Data, res1.Shape)
	fmt.Printf("Data: %s, Shape: %s\n", res2.Data, res2.Shape)

	// Output:
	// Data: [1 1 1 2 2 2], Shape: [2 3]
	// Data: [1 2 1 2], Shape: [2 2]
}

func ExampleMax() {
	node, _ := CreateGraph([]int{1, 32, 3, 4, 5, 3}, Shape{6})
	node, _ = Max(node)
	res, _ := CalculateResult[int](node)
	fmt.Println(res.Data[0])
	// Output: 32
}

func ExampleMin() {
	node, _ := CreateGraph([]int{1, 32, -3, 4, 5, 3}, Shape{6})
	node, _ = Min(node)
	res, _ := CalculateResult[int](node)
	fmt.Println(res.Data[0])
	// Output: -3
}

func ExampleOneHot() {
	node, _ := CreateGraph([]int{0, 1, 2, 1, 2, 0}, Shape{6})
	node, _ = OneHot(node, 3)
	res, _ := CalculateResult[int](node)
	fmt.Printf("Data: %s, Shape: %s\n", res.Data, res.Shape)
	// Output:
	// Data: [1 0 0 0 1 0 0 0 1 0 1 0 0 0 1 1 0 0], Shape: [6 3]
}
