package wrapper

// #include <flint/flint.h>
import "C"

/*
CalculateGradient Calculates the overall gradient of an output node to a variable.
The variable must be marked as a gradient variable, see [MarkGradientVariable]
and the output node must be constructed in a gradient context
(to remember which variables are directly or indirectly present in which operation),
which can be started with [StartGradientContext].
If you need to compute multiple gradients for one output use [CalculateGradients] since it is far more efficient.

Params:
  - [node]: the node which represents the chain of functions of which the gradient is to be computed.
  - [dx]: the variable for which node is derived for
*/
func CalculateGradient(node GraphNode, dx GraphNode) (GraphNode, error) {
	flintNode, errno := C.fCalculateGradient(node.ref, dx.ref)
	return returnHelper(flintNode, errno)
}

/*
CalculateGradients Calculates the overall gradient of an output node to multiple variables.
The variables must be marked as a gradient variable, see [MarkGradientVariable] and the output node must be constructed in a gradient context
(to remember which variables are directly or indirectly present in which operation),
which can be started with [StartGradientContext].

Params:
  - [node]: the node which represents the chain of functions of which the gradients are to be computed.
  - [dxs]: array of variables for which node is derived for.

Returns array with the same size as [dxs]
*/
func CalculateGradients(node GraphNode, dxs []GraphNode) ([]GraphNode, error) {
	n := len(dxs)

	partials := make([]*C.FGraphNode, n)
	for i, x := range dxs {
		partials[i] = x.ref
	}

	res := make([]*C.FGraphNode, n)

	errCode, errno := C.fCalculateGradients(node.ref, &(partials[0]), C.uint(n), &(res[0]))
	if err := buildErrorFromErrnoAndErrorType(errno, errorType(errCode)); err != nil {
		return []GraphNode{}, err
	}

	out := make([]GraphNode, n)
	for i, x := range res {
		out[i] = GraphNode{ref: x}
	}
	return out, nil
}

/*
StartGradientContext starts a gradient context. Gradient information will be inherited until the next call to [StopGradientContext].
A history containing information about all watched nodes in the parent graph is kept up to date within a gradient context for each node created in it.
The node does not have to be watched in that particular context (it just has to be marked as watched),
but only for operations which are constructed inside such a gradient context a gradient to a variable may be computed.
Since this context introduces overhead and limits [OptimizeMemory] significantly, it should be closed as soon as possible.
*/
func StartGradientContext() {
	C.fStartGradientContext()
}

/*
StopGradientContext stops a gradient context, all inherited gradient information and watched
nodes will be kept, but no longer inherited to new ones.
*/
func StopGradientContext() {
	C.fStopGradientContext()
}

/*
IsGradientContext returns true if the call to this function is placed within a gradient context.
*/
func IsGradientContext() bool {
	res := C.fIsGradientContext()
	return bool(res)
}

/*
MarkGradientVariable marks this [node] as a node for which a gradient might be calculated later.
It is only possible to calculate the gradient for this node (as a derivative) in operations that occur AFTER a call to this method.
All subsequent operations will have a remark that enlists this node as a possible gradient variable, to enable less memory usage and faster gradient calculation.
*/
func MarkGradientVariable(node GraphNode) {
	C.fMarkGradientVariable(node.ref)
}

/*
UnmarkGradientVariable Removes the gradient mark (and subsequent memory overhead) for this node.
After a call to this method NO subsequent gradient calculations with this node as a derivative will be possible.
*/
func UnmarkGradientVariable(node GraphNode) {
	C.fUnmarkGradientVariable(node.ref)
}
