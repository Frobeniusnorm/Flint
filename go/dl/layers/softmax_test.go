package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dl"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestNewSoftmax(t *testing.T) {
	l := NewSoftmax()
	assert.Empty(t, l)
}

func TestSoftmax_Forward(t *testing.T) {
	data := dl.NewTensor(flint.CreateGraphConstant(float32(2), flint.Shape{4}))
	softmax := NewSoftmax()
	out := softmax.Forward(data)
	res := flint.CalculateResult[float32](out.Node)
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
