package tensor

import "github.com/Frobeniusnorm/Flint/go/flint"

/*
Parameter is a container object for a trainable Tensor
*/
type Parameter[T Numeric] Tensor[T]

// NewParameter initializes the node as a trainable [Tensor]
func NewParameter[T Numeric](tensor Tensor[T]) Parameter[T] {
	if tensor.node == nil {
		// TODO: turn data into node
	}
	flint.MarkGradientVariable(*tensor.node)
	return Parameter[T](tensor)
}

func (p Parameter[T]) Close() {
	flint.UnmarkGradientVariable(*p.node)
	Tensor[T](p).Close()
}
