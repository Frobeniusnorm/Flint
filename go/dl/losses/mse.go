package losses

import (
	"github.com/Frobeniusnorm/Flint/go/dl"
	"github.com/Frobeniusnorm/Flint/go/flint"
	"log"
)

func MSELoss(prediction dl.Tensor, target dl.Tensor) dl.Tensor {
	return MSELossExtended(prediction, target, REDUCE_MEAN)
}

// MSELossExtended computes the MSE loss.
// This function includes all possible settings.
//
// param predictions: [dl.Tensor] of any shape
// param target: [dl.Tensor] with same shape as [predictions]
//
// returns: new tensor with a single value - the loss
func MSELossExtended(predictions dl.Tensor, target dl.Tensor, reduce reduction) dl.Tensor {
	shapePredictions := predictions.Node.GetShape()
	shapeTarget := target.Node.GetShape()
	if !shapePredictions.Equal(shapeTarget) {
		log.Panicf("shapes don't match: %s vs. %s", shapePredictions, shapeTarget)
	}
	N := shapePredictions[0] // batch size

	l := flint.Sub(predictions.Node, target.Node)
	l = flint.Mul(l, l)

	if reduce == REDUCE_SUM {
		l = flint.Sum(l)
	}
	if reduce == REDUCE_MEAN {
		l = flint.Divide(l, int32(N))
	}
	return dl.NewTensor(l)
}
