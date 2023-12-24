package losses

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint_old"
)

const eps float32 = 1e-10

// CrossEntropyLoss calculates the cross entropy loss between a set of two tensors
// predictions are the logits (thus one hot encoded). They do not have to be normalized, nor sum to one for this to work.
// target is the expected value, NOT one hot encoded
func CrossEntropyLoss(predictions flint_old.Tensor, target flint_old.Tensor) flint_old.Tensor {
	return CrossEntropyLossExtended(predictions, target, flint_old.Tensor{}, REDUCE_MEAN, 0.0)
}

/*
CrossEntropyLossExtended is the full implementation of the loss function.

param predictions: predictions are the logits (thus one hot encoded). They do not have to be normalized, nor sum to one for this to work
predictions has to be a Tensor of size (C) for unbatched input, (minibatch,C) for the typical logits or (minibatch,C,d1,d2,...,dK).
The last one might being useful for higher dimension inputs, such images.

param labels should be a flint_old with one of the following structures:
*/
func CrossEntropyLossExtended(predictions flint_old.Tensor, target flint_old.Tensor, weight flint_old.Tensor, reduce reduction, labelSmoothing float32) flint_old.Tensor {
	fmt.Println("Cross entropy loss extended")

	shape := predictions.Shape()
	//C := shape[1] // number of classes
	N := shape[0] // batch size

	// find out if the labels contain class indices (discrete) or probabilities (continuous)

	// TODO: take a look at wrapper.Equal
	// FIXME: everything basically lol (see pt)
	// FIXME: given one hot encoding we can also do: L(y, ŷ) = − log(ŷ_k)|y_k =1

	offset := flint_old.Constant(eps, shape)
	l := offset.Add(predictions).Log().Mul(target).Neg()
	for len(l.Shape()) > 1 {
		l = l.ReduceSum(0)
	}

	if reduce == REDUCE_SUM {
		l = l.Sum()
	}
	if reduce == REDUCE_MEAN {
		l = l.Div(flint_old.Scalar(int32(N)))
	}
	return l
}

// crossEntropyFromDiscrete calculates the cross entropy for discrete parameters, such as class labels
func crossEntropyFromDiscrete() {

}

// crossEntropyFromContinuous calculates the cross entropy for continuous parameters, such as probabilities
func crossEntropyFromContinuous() {

}
