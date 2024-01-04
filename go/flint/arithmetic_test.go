package flint

import (
	"fmt"
	"testing"
)

func TestAdd(t *testing.T) {
	x := Scalar(float32(3))
	fmt.Println("x", x)
	y := Random(Shape{2})
	fmt.Println("y", y)
	z := x.Add(y)
	fmt.Println("z", z)
}
