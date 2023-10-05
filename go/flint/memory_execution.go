package flint

// #include <flint/flint.h>
import "C"
import "unsafe"

func IncreaseRefCounter(node GraphNode) {
	var flintNode *C.FGraphNode = node.ref
	flintNode.reference_counter = C.size_t(flintNode.reference_counter + C.size_t(1))
}

func DecreaseRefCounter(node GraphNode) {
	var flintNode *C.FGraphNode = node.ref
	// make sure it does not underflow!
	if uint64(flintNode.reference_counter) == 0 {
		return // FIXME: throw error instead?
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
CopyGraph clones the graph node, the corresponding operation and additional data and the predecessors (their "GraphNode.reference_counter" is incremented)
*/
func CopyGraph(node GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fCopyGraph(node.ref)
	return GraphNode{ref: flintNode}
}

/*
ExecuteGraph executes the graph node operations from all yet to be executed predecessors to [node] and returns a [GraphNode] with a [ResultDataOld] operation in which the resulting data is stored.

If the graph is executed by the GPU backend, an OpenCL kernel containing all selected operations (the nodes operation and those indirect parent operations which were not yet executed) are compiled and executed.
The kernels are cached to improve performance of a program if the same graph-structures are reused (not necessary the same nodes, but the same combination of nodes), since then the backend can reuse already compiled kernels.
The GPU backend is allowed to keep the Result data on the GPU without synchronizing it (so the generated [ResultDataOld] may not have a "data" member!).
To force synchronization use [SyncMemory] or replace the call with [CalculateResult].

If the CPU backend is chosen, it does not matter, since every operation is executed independently (here eager execution might be faster).
The backend is selected by the framework if both are initialized, else the one that is initialized will be chosen.
If both are uninitialized, both will be initialized prior to execution.
If eager execution is enabled each node will be executed eagerly upon construction or with this method.

Also see [EnableEagerExecution], [SyncMemory]
*/
func ExecuteGraph(node GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fExecuteGraph(node.ref)
	return GraphNode{ref: flintNode}
}

/*
ExecuteGraphCpu executes the graph node operations from all yet to be executed predecessors to [node]
and returns a [GraphNode] with a [ResultDataOld] operation in which the resulting data is stored.
*/
func ExecuteGraphCpu(node GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fExecuteGraph_cpu(node.ref)
	return GraphNode{ref: flintNode}
}

/*
ExecuteGraphGpu executes the graph node operations from all yet to be executed predecessors to [node] and returns a [GraphNode]
with a [ResultDataOld] operation in which the resulting data is stored.
For the GPU backend, an opencl kernel containing all selected operations (the nodes operation and those indirect parent operations which were not yet executed) is compiled and executed.
The kernels are cashed, so it improves the performance of a program if the same graph-structures are reused (not necessary the same nodes, but the same combination of nodes),
since then the backend can reuse already compiled kernels.
*/
func ExecuteGraphGpu(node GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fExecuteGraph_gpu(node.ref)
	return GraphNode{ref: flintNode}
}

/*
ExecuteGraphCpuEagerly executes the graph node directly and assumes that predecessor data has already been computed.
Uses the CPU backend. Mainly used by helper functions of the framework, only use it if you now what you are doing.

Also see [EnableEagerExecution], [ExecuteGraphCpu] and [ExecuteGraph].
*/
func ExecuteGraphCpuEagerly(node GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fExecuteGraph_cpu_eagerly(node.ref)
	return GraphNode{ref: flintNode}
}

/*
ExecuteGraphGpuEagerly executes the graph node directly and assumes that predecessor data has already been computed.
Uses the GPU backend where for each operation and parameter type combination one eager kernel will be computed.
Mainly used by helper functions of the framework, only use it if you now what you are doing.

Also see [EnableEagerExecution], [ExecuteGraphGpu] and [ExecuteGraph].
*/
func ExecuteGraphGpuEagerly(node GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fExecuteGraph_gpu_eagerly(node.ref)
	return GraphNode{ref: flintNode}
}

/*
SyncMemory flushes all GPU data to the CPU.
[ExecuteGraph] does not guarantee that memory is present on the cpu (it may be kept on the GPU for performance reasons).
This method enforces all GPU data to be flushed to the CPU (but never executes the node!).
Also see [CalculateResult].
*/
func SyncMemory(node GraphNode) ResultData {
	res := C.fSyncMemory(node.ref)
	return ResultData{resultRef: res}
}

/*
CalculateResult is a convenience method that first executes [ExecuteGraph] and then [SyncMemory] on the [node].

It represents execution with one of both backends and additional memory synchronizing for the gpu framework.

FIXME: return [ResultDataOld] instead of tensor. Tensor should be moved to DL framework!
*/
func CalculateResult[T completeNumeric](node GraphNode) ResultData {
	var flintNode *C.FGraphNode = C.fCalculateResult(node.ref)

	dataSize := int(flintNode.result_data.num_entries)
	dataPtr := unsafe.Pointer(flintNode.result_data.data)
	shapePtr := unsafe.Pointer(flintNode.operation.shape)
	shapeSize := int(flintNode.operation.dimensions)

	dataType := DataType(flintNode.operation.data_type)

	var result = fromCToArray[T](dataPtr, dataSize, dataType)
	var shape = Shape(fromCToArray[uint](shapePtr, shapeSize, f_INT64))

	return ResultData{
		nodeRef:  flintNode,
		Data:     result,
		Shape:    shape,
		DataType: dataType,
	}
}

/*
CalculateGradient Calculates the overall gradient of an output node to a variable.
The variable must be marked as a gradient variable, see [MarkGradientVariable]
and the output node must be constructed in a gradient context
(to remember which variables are directly or indirectly present in which operation),
which can be started with [StartGradientContext].
If you need to compute multiple gradients for one output use [CalculateGradients] since it is far more efficient.

Params:
  - [node]: the Node which represents the chain of functions of which the gradient is to be computed.
  - [dx]: the variable for which node is derived for
*/
func CalculateGradient(node GraphNode, dx GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fCalculateGradient(node.ref, dx.ref)
	return GraphNode{ref: flintNode}
}

/*
CalculateGradients Calculates the overall gradient of an output node to multiple variables.
The variables must be marked as a gradient variable, see [MarkGradientVariable] and the output node must be constructed in a gradient context
(to remember which variables are directly or indirectly present in which operation),
which can be started with [StartGradientContext].

Params:
  - [node]: the Node which represents the chain of functions of which the gradients are to be computed.
  - [dxs]: array of variables for which node is derived for.

Returns array with the same size as [dxs]
*/
func CalculateGradients(node GraphNode, dxs []GraphNode) []GraphNode {
	n := len(dxs)

	partials := make([]*C.FGraphNode, n)
	for i, x := range dxs {
		partials[i] = x.ref
	}

	res := make([]*C.FGraphNode, n)

	C.fCalculateGradients(node.ref, &(partials[0]), C.uint(n), &(res[0]))

	out := make([]GraphNode, n)
	for i, x := range res {
		out[i] = GraphNode{ref: x}
	}
	return out
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

/*
OptimizeMemory frees all parental data (operand nodes of the operation of this node) and transforming this node to a storage nodes
if no gradient variables are present in this node and result data is present (i.e. it has been executed).
If you call this function manually please make sure to call [IncreaseRefCounter] of the parents if you want to keep their handles,
since this function may decrease their counter and free them.
*/
func OptimizeMemory(node GraphNode) GraphNode {
	var flintNode *C.FGraphNode = C.fOptimizeMemory(node.ref)
	return GraphNode{ref: flintNode}
}

/*
Serialize the data and shape of the node and returns an array of bytes, in which the serialized data will be written.
*/
func Serialize(node GraphNode) []byte {
	var size C.size_t
	ptr := C.fserialize(node.ref, &size)
	defer C.free(unsafe.Pointer(ptr))
	return C.GoBytes(unsafe.Pointer(ptr), C.int(size))
}

/*
Deserialize un-serializes data generated by [Serialize].
The size of the data is stored in itself, therefore no extra parameters are needed.
*/
func Deserialize(data []byte) GraphNode {
	unsafeData := C.CBytes(data)
	defer C.free(unsafe.Pointer(unsafeData))
	// this cast is necessary as the binary data needs to be passed as char*.
	var flintNode *C.FGraphNode = C.fdeserialize((*C.char)(unsafeData))
	return GraphNode{ref: flintNode}
}
