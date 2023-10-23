package layers

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/Frobeniusnorm/Flint/go/tensor"
	"testing"
)
import "github.com/stretchr/testify/assert"

func TestReLU_Forward(t *testing.T) {
	relu := NewRelu()

	input := flint.CreateGraphArrange(flint.Shape{5}, 0)
	input = flint.Sub(input, int64(3))

	output := relu.Forward(tensor.NewTensor(input))

	expected := []int64{0, 0, 0, 0, 1}

	res := flint.CalculateResult[int64](output.node)

	assert.EqualValues(t, expected, res.Data)
}

func TestReLU_Gradient(t *testing.T) {

}
