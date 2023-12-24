package losses

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

// NLLLoss computes the negative log likelihood loss.
// Best suited for classification problems.
// returns: a new flint with the result
func NLLLoss(x flint.Tensor) flint.Tensor {
	return x // TODO
}

func NLLLossExtended(x flint.Tensor, weight flint.Tensor, reduce reduction) flint.Tensor {
	shape := x.Shape()
	C := shape[1] // number of classes
	fmt.Println(C)
	return x // TODO
}
