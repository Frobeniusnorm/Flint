package losses

import (
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
	shapePredictions := predictions.Shape()
	shapeTarget := target.Shape()
	if !shapePredictions.Equal(shapeTarget) {
		log.Panicf("shapes don't match: %s vs. %s", shapePredictions, shapeTarget)
	}
	N := shapePredictions[0] // batch size

	l := predictions.Sub(target)
	l = l.Mul(l)

	if reduce == REDUCE_SUM {
		l = l.Sum()
	}
	if reduce == REDUCE_MEAN {
		l = l.Div(tensor.Scalar(int32(N)))
	}
	return l
}
