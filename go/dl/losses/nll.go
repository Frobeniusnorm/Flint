package losses

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/dl"
)

// NLLLoss computes the negative log likelihood loss.
// Best suited for classification problems.
// returns: a new tensor with the result
func NLLLoss(x dl.Tensor) dl.Tensor {
	return x // TODO
}

func NLLLossExtended(x dl.Tensor, weight dl.Tensor, reduce reduction) dl.Tensor {
	shape := x.Node.GetShape()
	C := shape[1] // number of classes
	fmt.Println(C)
	return x // TODO
}
