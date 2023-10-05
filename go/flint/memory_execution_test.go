package flint

import (
	"fmt"
	"testing"
)

// f_INT32 and so on has a bad relation to go types
// Just convert the entire input instead of trying to match types
func TestCalculateGradients(t *testing.T) {
	shape := Shape{2, 2}
	x1 := CreateGraphConstant(3, shape)
	x2 := CreateGraphConstant(4, shape)
	y1 := Mul(x1, x2)
	MarkGradientVariable(y1)
	partials := CalculateGradients(y1, []GraphNode{x1, x2})

	fmt.Println("result", CalculateResult[int32](y1))
	fmt.Println("dx1", CalculateResult[int32](partials[0]))
	fmt.Println("dx2", CalculateResult[int32](partials[1]))

	// TODO: assert partials[0] == x1
	// TODO: assert partials[1] == x2
}

func TestCalculateResult(t *testing.T) {

}
