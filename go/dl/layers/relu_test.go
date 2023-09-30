package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"testing"
)
import "github.com/stretchr/testify/assert"

func TestNewRelu(t *testing.T) {
	relu := NewRelu()
	assert.False(t, relu.trainable)
}

func TestReLU_Forward(t *testing.T) {
	relu := NewRelu()

	input := flint.CreateGraphArrange(flint.Shape{5}, 0)
	input = flint.Sub(input, int64(3))

	output := relu.Forward(Tensor{Node: input})

	expected := []int64{0, 0, 0, 0, 1}

	res := flint.CalculateResult[int64](output.Node)

	fmt.Println("expected:", expected)
	fmt.Println("output:", res.Data)

	assert.Equal(t, expected, res.Data)
}

func TestReLU_Gradient(t *testing.T) {

}
