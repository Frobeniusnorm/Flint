package losses

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint_old"
)

// NLLLoss computes the negative log likelihood loss.
// Best suited for classification problems.
// returns: a new flint_old with the result
func NLLLoss(x flint_old.Tensor) flint_old.Tensor {
	return x // TODO
}

func NLLLossExtended(x flint_old.Tensor, weight flint_old.Tensor, reduce reduction) flint_old.Tensor {
	shape := x.Shape()
	C := shape[1] // number of classes
	fmt.Println(C)
	return x // TODO
}
