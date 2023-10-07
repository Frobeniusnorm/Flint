package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dl"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSigmoid_Forward(t *testing.T) {
	data := dl.NewTensor(flint.CreateGraphConstant(float32(2), flint.Shape{4}))
	sigmoid := NewSigmoid()
	out := sigmoid.Forward(data)
	res := flint.CalculateResult[float32](out.Node)
	fmt.Println(res)
}

func TestSigmoid_Parameters(t *testing.T) {
	sigmoid := NewSigmoid()
	params := sigmoid.Parameters(false)
	assert.Len(t, params, 0)

	params = sigmoid.Parameters(true)
	assert.Len(t, params, 0)
}
