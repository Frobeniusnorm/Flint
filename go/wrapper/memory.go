package wrapper

// #include <flint/flint.h>
import "C"

func IncreaseRefCounter(node GraphNode) {
	var flintNode *C.FGraphNode = node.ref
	flintNode.reference_counter = C.size_t(flintNode.reference_counter + C.size_t(1))
}

func DecreaseRefCounter(node GraphNode) {
	var flintNode *C.FGraphNode = node.ref
	// make sure it does not underflow!
	if uint64(flintNode.reference_counter) == 0 {
		return
	}
	flintNode.reference_counter = C.size_t(flintNode.reference_counter - C.size_t(1))
}

func SetRefCounter(node GraphNode, value uint) {
	var flintNode *C.FGraphNode = node.ref
	flintNode.reference_counter = C.size_t(value)
}

/*
FreeGraph Decrements [node]'s "reference_counter" and deallocates the node and its corresponding data, if the counter reaches 0.
If the node is deallocated, the same process is repeated with its predecessors.
So you can safely connect nodes multiple times and have only to free the leaf nodes (i.e. the results),
without caring about cross-reference, since those are handled by the reference counting system.
*/
func FreeGraph(node GraphNode) {
	C.fFreeGraph(node.ref)
}

/*
SyncMemory flushes all GPU data to the CPU.
[ExecuteGraph] does not guarantee that memory is present on the cpu (it may be kept on the GPU for performance reasons).
This method enforces all GPU data to be flushed to the CPU (but never executes the node!).
Also see [CalculateResult].

TODO: return something?
*/
func SyncMemory(node GraphNode) {
	C.fSyncMemory(node.ref)
	//res := C.fSyncMemory(node.ref)
	//return Result{resultRef: res}
}

/*
OptimizeMemory frees all parental data (operand nodes of the operation of this node) and transforming this node to a storage nodes
if no gradient variables are present in this node and result data is present (i.e. it has been executed).
If you call this function manually please make sure to call [IncreaseRefCounter] of the parents if you want to keep their handles,
since this function may decrease their counter and free them.
*/
func OptimizeMemory(node GraphNode) (GraphNode, error) {
	flintNode, errno := C.fOptimizeMemory(node.ref)
	return returnHelper(flintNode, errno)
}

/*
EnforceInverseBroadcasting requires the given node to be inversely broadcasted.

Sometimes there are combinations of nodes where both normal and inverse
broadcasting is possible, but yields different results, e.g. multiplication
for two nodes with shapes [3, 5, 3, 5] and [3, 5]. The framework chooses
normal broadcasting over inverse if both are possible, this function allows
you to alter this behaviour and mark a node to be inversely broadcasted.
After the call to this function the given node will from then on only
inversely broadcasted (in cases where only normal broadcasting is available
an error will occur!). It has no effect in operations that don't use
broadcasting.
You can "unmark" the node with `fUnenforceInverseBroadcasting`.
*/
func EnforceInverseBroadcasting(node GraphNode) {
	C.fEnforceInverseBroadcasting(node.ref)
}

// UnEnforceInverseBroadcasting undoes the effect of [EnforceInverseBroadcasting] for a node.
func UnEnforceInverseBroadcasting(node GraphNode) {
	C.fUnenforceInverseBroadcasting(node.ref)
}
