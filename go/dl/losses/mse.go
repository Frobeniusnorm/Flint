package losses

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint_old"
	"log"
)

func MSELoss(prediction flint_old.Tensor, target flint_old.Tensor) flint_old.Tensor {
	return MSELossExtended(prediction, target, REDUCE_MEAN)
}

// MSELossExtended computes the MSE loss.
// This function includes all possible settings.
//
// param predictions: [dl.Tensor] of any shape
// param target: [dl.Tensor] with same shape as [predictions]
//
// returns: new flint_old with a single value - the loss
func MSELossExtended(predictions flint_old.Tensor, target flint_old.Tensor, reduce reduction) flint_old.Tensor {
	fmt.Println("MSELossExtended")
	shapePredictions := predictions.Shape()
	shapeTarget := target.Shape()
	if !shapePredictions.Equal(shapeTarget) {
		log.Panicf("shapes don't match: %s vs. %s", shapePredictions, shapeTarget)
	}
	N := shapePredictions[0] // batch size

	x := predictions.Sub(target)
	x = x.Mul(x)

	if reduce == REDUCE_SUM {
		x = x.Sum()
	}
	if reduce == REDUCE_MEAN {
		fmt.Println("x shape", x.Shape().String())
		x = x.Div(flint_old.Scalar(int32(N)))
	}
	return x
}
