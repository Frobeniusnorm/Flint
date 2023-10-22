package tensor

import "github.com/Frobeniusnorm/Flint/go/flint"

/*
Parameter is a container object for a trainable Tensor
*/
type Parameter Tensor

// NewParameter initializes the node as a trainable [Tensor]
func NewParameter(node flint.GraphNode) Parameter {
	flint.MarkGradientVariable(node)
	tensor := NewTensor(node)
	return Parameter(tensor)
}

func (p Parameter) Close() {
	flint.UnmarkGradientVariable(p.Node)
	Tensor(p).Close()
}
