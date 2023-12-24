package flint_old

import "github.com/Frobeniusnorm/Flint/go/wrapper"

/*
Parameter is a container object for a trainable Tensor
*/
type Parameter Tensor

// NewParameter initializes the node as a trainable [Tensor]
func NewParameter(tensor Tensor) Parameter {
	if tensor.isLight() {
		tensor = tensor.readyNode()
	}
	wrapper.MarkGradientVariable(*tensor.node)
	return Parameter(tensor)
}

func (p Parameter) Close() {
	wrapper.UnmarkGradientVariable(*p.node)
	Tensor(p).Close()
}

func (p Parameter) Node() wrapper.GraphNode {
	// Parameter constructor makes sure that node is already ready
	return *p.node
}
