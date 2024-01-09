package wrapper

// #include <flint/flint.h>
import "C"

/*
ExecuteGraph executes the graph node operations from all yet to be executed predecessors to [node] and returns a [GraphNode] with a [ResultDataOld] operation in which the resulting data is stored.

If the graph is executed by the GPU backend, an OpenCL kernel containing all selected operations (the nodes operation and those indirect parent operations which were not yet executed) are compiled and executed.
The kernels are cached to improve performance of a program if the same graph-structures are reused (not necessary the same nodes, but the same combination of nodes), since then the backend can reuse already compiled kernels.
The GPU backend is allowed to keep the result data on the GPU without synchronizing it (so the generated [ResultDataOld] may not have a "data" member!).
To force synchronization use [SyncMemory] or replace the call with [CalculateResult].

If the CPU backend is chosen, it does not matter, since every operation is executed independently (here eager execution might be faster).
The backend is selected by the framework if both are initialized, else the one that is initialized will be chosen.
If both are uninitialized, both will be initialized prior to execution.
If eager execution is enabled each node will be executed eagerly upon construction or with this method.

Also see [SetEagerExecution], [SyncMemory]
*/
func ExecuteGraph(node GraphNode) (GraphNode, error) {
	flintNode, errno := C.fExecuteGraph(node.ref)
	return returnHelper(flintNode, errno)
}

/*
ExecuteGraphCpu executes the graph node operations from all yet to be executed predecessors to [node]
and returns a [GraphNode] with a [ResultDataOld] operation in which the resulting data is stored.
*/
func ExecuteGraphCpu(node GraphNode) (GraphNode, error) {
	flintNode, errno := C.fExecuteGraph_cpu(node.ref)
	return returnHelper(flintNode, errno)
}

/*
ExecuteGraphGpu executes the graph node operations from all yet to be executed predecessors to [node] and returns a [GraphNode]
with a [ResultDataOld] operation in which the resulting data is stored.
For the GPU backend, an opencl kernel containing all selected operations (the nodes operation and those indirect parent operations which were not yet executed) is compiled and executed.
The kernels are cashed, so it improves the performance of a program if the same graph-structures are reused (not necessary the same nodes, but the same combination of nodes),
since then the backend can reuse already compiled kernels.
*/
func ExecuteGraphGpu(node GraphNode) (GraphNode, error) {
	flintNode, errno := C.fExecuteGraph_gpu(node.ref)
	return returnHelper(flintNode, errno)
}

/*
ExecuteGraphCpuEagerly executes the graph node directly and assumes that predecessor data has already been computed.
Uses the CPU backend. Mainly used by helper functions of the framework, only use it if you now what you are doing.

Also see [SetEagerExecution], [ExecuteGraphCpu] and [ExecuteGraph].
*/
func ExecuteGraphCpuEagerly(node GraphNode) (GraphNode, error) {
	flintNode, errno := C.fExecuteGraph_cpu_eagerly(node.ref)
	return returnHelper(flintNode, errno)
}

/*
ExecuteGraphGpuEagerly executes the graph node directly and assumes that predecessor data has already been computed.
Uses the GPU backend where for each operation and parameter type combination one eager kernel will be computed.
Mainly used by helper functions of the framework, only use it if you now what you are doing.

Also see [SetEagerExecution], [ExecuteGraphGpu] and [ExecuteGraph].
*/
func ExecuteGraphGpuEagerly(node GraphNode) (GraphNode, error) {
	flintNode, errno := C.fExecuteGraph_gpu_eagerly(node.ref)
	return returnHelper(flintNode, errno)
}

/*
CalculateResult is a convenience method that first executes [ExecuteGraph] and then [SyncMemory] on the [node].

It represents execution with one of both backends and additional memory synchronizing for the gpu framework.
*/
func CalculateResult(node GraphNode) (GraphNode, error) {
	flintNode, errno := C.fCalculateResult(node.ref)
	return returnHelper(flintNode, errno)
}
