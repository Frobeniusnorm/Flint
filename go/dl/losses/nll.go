package losses

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/tensor"
)

// NLLLoss computes the negative log likelihood loss.
// Best suited for classification problems.
// returns: a new tensor with the result
func NLLLoss(x tensor.Tensor) tensor.Tensor {
	return x // TODO
}

func NLLLossExtended(x tensor.Tensor, weight tensor.Tensor, reduce reduction) tensor.Tensor {
	shape := x.node.GetShape()
	C := shape[1] // number of classes
	fmt.Println(C)
	return x // TODO
}
