package losses

import (
	"github.com/Frobeniusnorm/Flint/go/flint"
	"github.com/Frobeniusnorm/Flint/go/tensor"
	"log"
)

func MSELoss(prediction tensor.Tensor, target tensor.Tensor) tensor.Tensor {
	return MSELossExtended(prediction, target, REDUCE_MEAN)
}

// MSELossExtended computes the MSE loss.
// This function includes all possible settings.
//
// param predictions: [dl.Tensor] of any shape
// param target: [dl.Tensor] with same shape as [predictions]
//
// returns: new tensor with a single value - the loss
func MSELossExtended(predictions tensor.Tensor, target tensor.Tensor, reduce reduction) tensor.Tensor {
	shapePredictions := predictions.node.GetShape()
	shapeTarget := target.node.GetShape()
	if !shapePredictions.Equal(shapeTarget) {
		log.Panicf("shapes don't match: %s vs. %s", shapePredictions, shapeTarget)
	}
	N := shapePredictions[0] // batch size

	l := flint.Sub(predictions.node, target.node)
	l = flint.Mul(l, l)

	if reduce == REDUCE_SUM {
		l = flint.Sum(l)
	}
	if reduce == REDUCE_MEAN {
		l = flint.Divide(l, int32(N))
	}
	return tensor.NewTensor(l)
}
