package layers

import (
	"fmt"
	"github.com/Frobeniusnorm/Flint/go/flint"
)

type Layer interface {
	fmt.Stringer

	// Parameters returns a list of all tensors in the layer which should be tweaked by the optimizer.
	// If the parameter [recurse] is set, it will call [Parameters] on all submodules as well (if there are any).
	Parameters(recurse bool) []flint.Parameter

	// Forward run the inference of the layer
	Forward(x flint.Tensor) flint.Tensor

	// TrainMode puts the layer in training mode.
	// This is just a feature toggle to change the behaviour of certain layers in the different phases of the pipeline (e.g Dropout)
	// Changing this mode has no effect on the gradients whatsoever.
	TrainMode()

	// EvalMode puts the layer in eval mode.
	// This is just a feature toggle to change the behaviour of certain layers in the different phases of the pipeline (e.g Dropout)
	// Changing this mode has no effect on the gradients whatsoever.
	EvalMode()
}
