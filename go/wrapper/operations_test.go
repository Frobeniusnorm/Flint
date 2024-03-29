package wrapper

import "fmt"

func ExampleReduceMax() {
	node1, _ := CreateGraphDataInt([]Int{1, 32, 3, 4, 5, 3}, Shape{2, 3})
	node2, _ := CreateGraphDataInt([]Int{9, 2, 3, -1, 5, 6}, Shape{2, 3})

	node1, _ = ReduceMax(node1, 0)
	node2, _ = ReduceMax(node2, 1)

	node1, _ = CalculateResult(node1)
	node2, _ = CalculateResult(node2)

	res1 := GetResultValue[Int](node1)
	res2 := GetResultValue[Int](node2)

	fmt.Printf("Data: %v, Shape: %s\n", res1, node1.GetShape())
	fmt.Printf("Data: %v, Shape: %s\n", res2, node2.GetShape())

	// Output:
	// Data: [4 32 3], Shape: [3]
	// Data: [9 6], Shape: [2]
}

func ExampleExpand() {
	node1, _ := CreateGraphDataInt([]Int{1, 2}, Shape{2})
	node2 := node1

	node1, _ = Expand(node1, uint(1), 3)
	node2, _ = Expand(node2, uint(0), 2)

	node1, _ = CalculateResult(node1)
	node2, _ = CalculateResult(node2)

	res1 := GetResultValue[Int](node1)
	res2 := GetResultValue[Int](node2)

	fmt.Printf("Data: %v, Shape: %s\n", res1, node1.GetShape())
	fmt.Printf("Data: %v, Shape: %s\n", res2, node2.GetShape())

	// Output:
	// Data: [1 1 1 2 2 2], Shape: [2 3]
	// Data: [1 2 1 2], Shape: [2 2]
}
