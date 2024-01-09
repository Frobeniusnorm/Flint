package flint

import (
	"fmt"
	"runtime"
	"testing"
)

func TestTensor_Close(t *testing.T) {
	x := Random(Shape{1})
	x = Random(Shape{2})
	runtime.GC()
	fmt.Println(x.Add(Scalar(int32(5))))
}
