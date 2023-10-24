package tensor

import (
	"fmt"
	"testing"
)

func TestTensor_Sub(t *testing.T) {
	x := Scalar(int32(2))
	fmt.Println(x)
	y := Random(Shape{2})
	fmt.Println(y)
	x2 := x.ToFloat64() // FIXME: this is insanely slow in debugger??
	fmt.Println(x2)
	z := x2.Sub(y)
	fmt.Println(z)
}
