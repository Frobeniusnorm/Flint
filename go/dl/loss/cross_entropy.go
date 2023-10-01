package loss

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

const eps float32 = 1e-10

// CrossEntropyLoss calculates the cross entropy loss between a set of two tensors
// TODO: for now: one hot encoding is implicitly assumed!
func CrossEntropyLoss(predictions flint.GraphNode, labels flint.GraphNode) flint.GraphNode {

	fmt.Println("pred shape:", predictions.GetShape())
	fmt.Println("labels shape:", labels.GetShape())

	shape := predictions.GetShape()
	if !shape.Equal(labels.GetShape()) {
		panic("shapes do not match")
	}

	// FIXME: given one hot encoding we can also do: L(y, ŷ) = − log(ŷ_k)|y_k =1

	offset := flint.CreateGraphConstant(eps, shape, flint.F_FLOAT32)
	t := flint.Neg(flint.Mul(flint.Log(flint.Add(offset, predictions)), labels))
	for len(t.GetShape()) > 1 {
		t = flint.ReduceSum(t, 0)
	}
	t = flint.ReduceSum(t, 0)
	flint.IncreaseRefCounter(t)
	return t
}
