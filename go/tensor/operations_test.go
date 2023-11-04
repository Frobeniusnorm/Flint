package tensor

import (
	"fmt"
	"testing"
)

func TestTensor_Sub(t *testing.T) {
	//x := Scalar(float32(2.3))
	x := Scalar(int32(3))
	fmt.Println("x", x)
	y := Random(Shape{2})
	fmt.Println("y", y)
	//x = x.To(flint.F_INT32) // FIXME: this is insanely slow in debugger??
	//fmt.Println("new x", x)
	z := x.Add(y)
	fmt.Println("z", z)
}
