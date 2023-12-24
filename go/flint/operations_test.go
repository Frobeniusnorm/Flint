package flint

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
	"testing"
)

func TestTensor_Sub(t *testing.T) {
	x := Scalar(float32(3))
	fmt.Println("x", x)
	y := Random(Shape{2})
	fmt.Println("y", y)
	x = x.To(wrapper.F_INT32)
	fmt.Println("new x", x)
	z := x.Add(y)
	fmt.Println("z", z)
}
