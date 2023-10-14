package losses

import (
	"github.com/Frobeniusnorm/Flint/go/dl"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMSELoss(t *testing.T) {
	predictions := dl.NewTensor(flint.CreateScalar(5))
	target := dl.NewTensor(flint.CreateScalar(2))
	loss := MSELoss(predictions, target)
	res := flint.CalculateResult[int](loss.Node).Data[0]

	assert.Equal(t, 9, res)
}
