package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/Frobeniusnorm/Flint/go/tensor"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestNewSigmoid(t *testing.T) {
	l := NewSigmoid()
	assert.Empty(t, l)
}

func TestSigmoid_Forward(t *testing.T) {
	data := tensor.NewTensor(flint.CreateGraphConstant(float32(2), flint.Shape{4}))
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
