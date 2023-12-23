package losses

import (
	"fmt"
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
		x = x.Div(tensor.Scalar(int32(N)))
	}
	return x
}
