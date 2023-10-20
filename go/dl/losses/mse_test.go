package losses

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/Frobeniusnorm/Flint/go/tensor"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMSELoss(t *testing.T) {
	predictions := tensor.NewTensor(flint.CreateScalar(5))
	target := tensor.NewTensor(flint.CreateScalar(2))
	loss := MSELoss(predictions, target)
	res := flint.CalculateResult[int](loss.Node).Data[0]

	assert.Equal(t, 9, res)
}
