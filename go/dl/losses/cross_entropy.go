package losses

import (
	"github.com/Frobeniusnorm/Flint/go/dl"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

const eps float32 = 1e-10

// CrossEntropyLoss calculates the cross entropy loss between a set of two tensors
// predictions are the logits (thus one hot encoded). They do not have to be normalized, nor sum to one for this to work.
// target is the expected value, NOT one hot encoded
func CrossEntropyLoss(predictions dl.Tensor, target dl.Tensor) dl.Tensor {
	return CrossEntropyLossExtended(predictions, target, dl.Tensor{}, REDUCE_MEAN, 0.0)
}

/*
CrossEntropyLossExtended is the full implementation of the loss function.

param predictions: predictions are the logits (thus one hot encoded). They do not have to be normalized, nor sum to one for this to work
predictions has to be a Tensor of size (C) for unbatched input, (minibatch,C) for the typical logits or (minibatch,C,d1,d2,...,dK).
The last one might being useful for higher dimension inputs, such images.

param labels should be a tensor with one of the following structures:
*/
func CrossEntropyLossExtended(predictions dl.Tensor, target dl.Tensor, weight dl.Tensor, reduce reduction, labelSmoothing float32) dl.Tensor {
	shape := predictions.Node.GetShape()
	//C := shape[1] // number of classes
	N := shape[0] // batch size

	// find out if the labels contain class indices (discrete) or probabilities (continuous)

	// TODO: take a look at flint.Equal
	// FIXME: everything basically lol (see pt)
	// FIXME: given one hot encoding we can also do: L(y, ŷ) = − log(ŷ_k)|y_k =1

	offset := flint.CreateGraphConstant(eps, shape)
	l := flint.Neg(flint.Mul(flint.Log(flint.Add(offset, predictions.Node)), target.Node))
	for len(l.GetShape()) > 1 {
		l = flint.ReduceSum(l, 0)
	}

	if reduce == REDUCE_SUM {
		l = flint.Sum(l)
	}
	if reduce == REDUCE_MEAN {
		l = flint.Divide(l, int32(N))
	}
	return dl.NewTensor(l)
}

// crossEntropyFromDiscrete calculates the cross entropy for discrete parameters, such as class labels
func crossEntropyFromDiscrete() {

}

// crossEntropyFromContinuous calculates the cross entropy for continuous parameters, such as probabilities
func crossEntropyFromContinuous() {

}
