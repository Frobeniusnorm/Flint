package flint

import (
	"fmt"
	"testing"
)

// FIXME: need to change all the type casting and checking.
// F_INT32 and so on has a bad relation to go types
// Just convert the entire input instead of trying to match types
func TestCalculateGradients(t *testing.T) {
	shape := Shape{2, 2}
	x1 := CreateGraphConstant(1, shape, F_INT32)
	x2 := CreateGraphConstant(2, shape, F_INT32)
	y1 := Add(x1, x2)
	MarkGradientVariable(y1)
	partials := CalculateGradients(y1, []GraphNode{x1, x2})
	fmt.Println(partials)
	// FIXME: print them
}
