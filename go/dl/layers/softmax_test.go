package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint_old"
	"github.com/Frobeniusnorm/Flint/go/wrapper"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestNewSoftmax(t *testing.T) {
	l := NewSoftmax()
	assert.Empty(t, l)
}

func TestSoftmax_Forward(t *testing.T) {
	data := flint_old.NewTensor(wrapper.CreateGraphConstant(float32(2), wrapper.Shape{4}))
	softmax := NewSoftmax()
	out := softmax.Forward(data)
	res := wrapper.CalculateResult[float32](out.node)
	fmt.Println(res.Data)

	var sum float32 = 0.0
	for _, val := range res.Data {
		sum += val
	}

	assert.Equal(t, float32(1.0), sum)
}

func TestSoftmax_Parameters(t *testing.T) {
	softmax := NewSoftmax()
	params := softmax.Parameters(false)
	assert.Len(t, params, 0)

	params = softmax.Parameters(true)
	assert.Len(t, params, 0)
}
