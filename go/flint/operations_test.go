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

func ExampleMax() {
	node := CreateGraph([]int{1, 32, 3, 4, 5, 3}, Shape{6})
	maximum := Max[int](node)
	fmt.Println(maximum)
	// Output: 32
}

func ExampleMin() {
	node := CreateGraph([]int{1, 32, -3, 4, 5, 3}, Shape{6})
	maximum := Min[int](node)
	fmt.Println(maximum)
	// Output: -3
}
