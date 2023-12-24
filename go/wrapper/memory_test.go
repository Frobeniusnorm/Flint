package wrapper

import (
	"testing"
)

// INT32 and so on has a bad relation to go types
// Just convert the entire input instead of trying to match types
//func TestCalculateGradients(t *testing.T) {
//	shape := Shape{2, 2}
//	x1, _ := CreateGraphConstant(3, shape)
//	x2, _ := CreateGraphConstant(4, shape)
//	y1, _ := Mul(x1, x2)
//	MarkGradientVariable(y1)
//	MarkGradientVariable(x1)
//	MarkGradientVariable(x2)
//	partials, err := CalculateGradients(y1, []GraphNode{x1, x2})
//
//	assert.NoError(t, err)
//
//	result, _ := CalculateResult[int32](y1)
//	dx1, _ := CalculateResult[int32](partials[0])
//	dx2, _ := CalculateResult[int32](partials[1])
//
//	fmt.Println("result", result)
//	fmt.Println("dx1", dx1)
//	fmt.Println("dx2", dx2)
//
//	// TODO: assert partials[0] == x1
//	// TODO: assert partials[1] == x2
//}

func TestCalculateResult(t *testing.T) {

}
