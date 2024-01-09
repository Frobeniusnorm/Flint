package flint

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestCreateFromSlice(t *testing.T) {
	x := [][]Int{
		{1, 2, 5, 5},
		{1, 2, 3, 3},
	}
	tens := CreateFromSlice(x, wrapper.INT32)
	assert.Equal(t, tens.Shape(), Shape{8})
	fmt.Println(tens)
}
