package tensor

import "github.com/Frobeniusnorm/Flint/go/flint"

/*
Parameter is a container object for a trainable Tensor
*/
type Parameter Tensor

// NewParameter initializes the node as a trainable [Tensor]
func NewParameter(tensor Tensor) Parameter {
	if tensor.node == nil {
		// TODO: turn data into node
	}
	flint.MarkGradientVariable(*tensor.node)
	return Parameter(tensor)
}

func (p Parameter) Close() {
	flint.UnmarkGradientVariable(*p.node)
	Tensor(p).Close()
}
