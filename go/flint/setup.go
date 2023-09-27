package flint

// #include <flint/flint.h>
import "C"

type Backend int

const (
	BACKEND_ONLY_CPU Backend = iota + 1
	BACKEND_ONLY_GPU
	BACKEND_BOTH
)

/*
Init initializes the backends.
These functions are already implicitly called by the execution functions if necessary.
The method allows disabling of the GPU backend by passing [BACKEND_ONLY_CPU].
However, this is not recommended since Flint usually chooses the right backend with heuristics.

Only use this function if you...
  - ...want to explicitly decide where and when the initialization should take place
  - ...want to only start one backend
*/
func Init(backend Backend) {
	C.flintInit(C.int(backend))
}

// InitializedBackend returns the initialized [Backend].
func InitializedBackend() Backend {
	// FIXME: this might break when a new backend is added,
	// as this go frontend does not doe the bit magic
	return Backend(C.flintInitializedBackends())
}

/*
Cleanup deallocates any resourced allocated by the corresponding backends.
This method calls the other two (following) which are only executed if the framework was initialized, else they do nothing.
*/
func Cleanup() {
	C.flintCleanup()
}

/*
CleanupCpu deallocates any resourced allocated by the CPU backend, if it was initialized, else it does nothing.
*/
func CleanupCpu() {
	C.flintCleanup_cpu()
}

/*
CleanupGpu deallocates any resourced allocated by the GPU backend, if it was initialized, else it does nothing.
*/
func CleanupGpu() {
	C.flintCleanup_gpu()
}

/*
SetEagerExecution changes the execution paradigm.

When set to true, all graph nodes that represent actual operations are after this call executed eagerly, i.e. they are executed during graph construction.
This may improve performance when only using the CPU backend, in any other case disabling eager execution should be preferred.

Disable eager execution, i.e. the graph is constructed without execution of the nodes until an operation makes the execution of a parent graph necessary
or the user calls [ExecuteGraph].
*/
func SetEagerExecution(turnOn bool) {
	if turnOn == true {
		C.fEnableEagerExecution()
	} else {
		C.fDisableEagerExecution()
	}
}

// IsEagerExecution returns true if eager execution has been enabled, else false
func IsEagerExecution() bool {
	res := int(C.fIsEagerExecution())
	if res == 1 {
		return true
	} else {
		return false
	}
}
