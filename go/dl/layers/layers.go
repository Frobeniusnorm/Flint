package layers

import "fmt"

type BaseLayer struct {
	trainable  bool
	EnableGrad bool // Essentially the same as "training_phase". Grads are disabled while testing.
}

type Layer interface {
	fmt.Stringer
	Forward(x Tensor) Tensor
	//Backward(x Tensor) Tensor
}

/*
LayerList is a storage container for layers.
Just passing them around may not create the EnableGrad flag properly

What's the difference between a [Sequential] and a [LayerList]?
A [LayerList] is exactly what it sounds like -- a list for storing [Layer]'s!
On the other hand, the layers in a [Sequential] are connected in a cascading way.
*/
type LayerList struct {
}

/*
Parameter is a container object for a trainable Tensor
*/
type Parameter Tensor
