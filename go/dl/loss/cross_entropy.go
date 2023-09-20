package loss

import flint "github.com/Frobeniusnorm/Flint/go"

const eps float32 = 1e-12

// CrossEntropyLoss calculates the cross entropy loss between a set of two tensors
// TODO: for now: one hot encoding is implicitly assumed!
func CrossEntropyLoss(predictions flint.GraphNode, labels flint.GraphNode) flint.GraphNode {
	shape := flint.GetShape(predictions)
	if !shape.Equal(flint.GetShape(labels)) {
		panic("shapes do not match")
	}

	offset := flint.CreateGraphConstant(eps, shape)
	t := flint.Neg(flint.Mul(flint.Log(flint.Add(offset, predictions)), labels))
	for len(flint.GetShape(t)) > 1 {
		t = flint.ReduceSum(t, 0)
	}
	t = flint.ReduceSum(t, 0)
	flint.IncreaseRefCounter(t)
	return t
}
