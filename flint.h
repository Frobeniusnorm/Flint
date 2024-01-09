/* Copyright 2023 David Schwarzbeck
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#ifndef FLINT_H
#define FLINT_H
#define CL_TARGET_OPENCL_VERSION 200

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif // __APPLE__

#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/* \file flint.h
  \brief This is the basic header file and implementation of Flint, written in C
  to be as compatile as possible

  Flints Execution structure represents an AST, so that each Graph may be
  compiled to a specific OpenCL program.

  To support C++ exceptions as well as keep C-compatibility there is a flag
  C_COMPATIBILITY that - when enabled during compilation - disables exception
  throwing and sets the errno instead (the functions then return NULL on error).
  You can query the error reason with `fErrorMessage` and `fErrorType`.

  In general all operations that take two parameters of equal shape (like e.g.
  addition, division, minimum, equal etc.) allow normal and inverse
  broadcasting.
  - normal broadcasting: a node with shape [4, 6, 8] can be broadcasted to a
	node with shape [2, 4, 6, 8] by repeating the first node 2 times in the
	first dimension.
  - inverse broadcasting: a node with shape [2, 4, 6] can be broadcasted to a
	node with shape [2, 4, 6, 8] by repeating the first node 8 times in the last
	dimension.
  E.g.

  @code{
  float data_a[] = {0, 1, 2,
					3, 4, 5};
  size_t shape_a[] = {2, 3};
  float data_b[] = {2, 4, 6};
  size_t shape_b = 3;
  FGraphNode* a = fCreateGraph((void*)data_a, 6, F_FLOAT32, shape_a, 2);
  FGraphNode* b = fCreateGraph((void*)data_b, 3, F_FLOAT32, &shape_b, 1);
  FGraphNode* c = fmul(a, b); // {{2, 5, 8}, {5, 8, 11}}
  }

  Broadcasting is implemented without repeating the data, but by directly
  accessing it.
*/
#define FLINT_BACKEND_ONLY_CPU 1
#define FLINT_BACKEND_ONLY_GPU 2
#define FLINT_BACKEND_BOTH 3

/** Types of erros that can occur in the framework (also see `fErrorMessage`)
 * Error Types:
 * - `NO_ERROR`: no error occured up until now
 * - `WRONG_TYPE`: Tensor has wrong data type, e.g. a floating point tensor in
 *   `feven`
 * - `ILLEGAL_DIMENSION`: the dimension parameter is not inside the legal range
 *   of the function, e.g. flattening the first dimension of a tensor.
 * - `ILLEGAL_DIMENSIONALITY`: the dimensionality of a parameter does not work
 *   with the function, usually because it is too low (e.g. matrix
 *   multiplication with 1 dimensional tensors).
 * - `INCOMPATIBLE_SHAPES`: the shapes of the parameters dont fit together
 * - `INVALID_SELECT`: a index or slicing operation received parameters which
 *   are semantically impossible or outside of the shape of the tensor.
 * - `OCL_ERROR`: OpenCL Error
 * - `INTERNAL_ERROR`: illegal state of flint, should not happen
 * - `OUT_OF_MEMORY`: no more cpu or gpu memory available
 * - `ILLEGAL_DERIVE`: derivation of graph to a variable is not possible
 * - `IO_ERROR`: file writing or reading problem
 */
enum FErrorType {
	NO_ERROR,
	WRONG_TYPE,
	ILLEGAL_DIMENSION,
	ILLEGAL_DIMENSIONALITY,
	INCOMPATIBLE_SHAPES,
	INVALID_SELECT,
	OCL_ERROR,
	INTERNAL_ERROR, // if this happens -> bug in code
	OUT_OF_MEMORY,
	ILLEGAL_DERIVE,
	IO_ERROR
};
typedef enum FErrorType FErrorType;

/** Initializes the cpu and the gpu backends. These functions are already
 * implicitly called by the execution functions if necessary. The method allows
 * disabling of the gpu backend (by passing `FLINT_BACKEND_ONLY_CPU`), disabling
 * of the cpu backend (by passing `FLINT_BACKEND_BOTH`), initializing both
 * backends explicitly (by passing `FLINT_BACKEND_BOTH`, which is recommended,
 * since Flint is then allowed to choose the framework with heuristics). Only
 * use those functions if you...
 * - ...want to explicitly decide where and when the initialization should take
 *      place
 * - ...want to only start one backend
 *
 * Returns `NO_ERROR` on success or the error type
 * */
FErrorType flintInit(int backends);

/** Don't call this function explicitly if you intent to use Flint normally. Use
 * `flintInit`. Returns `NO_ERROR` on success or the error type */
FErrorType flintInit_cpu();

/** Don't call this function explicitly if you intent to use Flint normally. Use
 * `flintInit`. Returns `NO_ERROR` on success or the error type */
FErrorType flintInit_gpu();

/** Returns an integer containing the Backend information bitwise.
 * See constants `FLINT_BACKEND_ONLY_CPU`, `FLINT_BACKEND_ONLY_GPU` and
 * `FLINT_BACKEND_BOTH`. */
int flintInitializedBackends();

/** Deallocates any resourced allocated by the corresponding backends.
This method calls the other two (following) which are only executed if the
framework was initialized, else they do nothing. Returns `NO_ERROR` on success
or the error type */
FErrorType flintCleanup();

/** Deallocates any resourced allocated by the cpu backend, if it was
 * initialized, else it does nothing. Returns `NO_ERROR` on success or the error
 * type */
FErrorType flintCleanup_cpu();

/** Deallocates any resourced allocated by the gpu backend, if it was
 * initialized, else it does nothing. Returns `NO_ERROR` on success or the error
 * type */
FErrorType flintCleanup_gpu();

/**
 * See also: `flogging`, `FLogType`
 * - `F_DEBUG` (only internal debugging informations of the framework),
 * - `F_VERBOSE` (verbose information, may be helpful to users of the
 *    library),
 * - `F_INFO` (informational data of the framework, e.g. which
 *    graphics card has been chosen),
 * - `F_ERROR` (unrecoverable errors,
 *    generated by function calls to the framework, raises a exception
 *    everytime),
 * - `F_WARNING` (probably unwanted behaviour or undefined behaviour caused
 *    by missuse of functions).
 */
enum FLogType { F_NO_LOGGING, F_ERROR, F_WARNING, F_INFO, F_VERBOSE, F_DEBUG };

/** Sets the logging level of the framework. Adjust this for debugging purposes,
 * or if you release software in which Flint is contained.
 * See also: `flogging`, `FLogType`
 *
 * Levels:
 * - 0: No logging
 * - 1: Only `F_ERROR`
 * - 2: Logging level `F_WARNING` (should be used for production)
 * - 3: Logging level `F_INFO` (for developement)
 * - 4: Logging level `F_VERBOSE` (for library developement)
 * - 5: Logging level `F_DEBUG` (when a bug in the library has been found)
 */
void fSetLoggingLevel(enum FLogType type);

/** Supported Image formats for fstore_image */
enum FImageFormat { F_PNG, F_JPEG, F_BMP };

/** Logs a NULL terminated string with the given logging level.
 * See also: `fSetLoggingLevel` */
void flogging(enum FLogType type, const char *msg);

/**
 * Queries the type of the last error that occured in this framework.
 * Errors cause an exception or if C-compatibility is enabled the function in
 * which the error occurs returns NULL. Then this function yields the type of
 * error.
 */
FErrorType fErrorType();

/**
 * Queries the message of the last error that occured in this framework.
 * Errors cause an exception or if C-compatibility is enabled the function in
 * which the error occurs returns NULL. Then this function yields the message
 * of the error. If no error occured returns an empty string.
 */
const char *fErrorMessage();

/** All graph nodes that represent actual operations are after this call
 * executed eagerly, i.e. they are executed during graph construction.
 *
 * This may improve performance when only using the CPU backend, in any other
 * case disabling eager execution should be preferred. */
void fEnableEagerExecution();

/** Disable eager execution, i.e. the graph is constructed without execution of
 * the nodes until an operation makes the execution of a parent graph necessary
 * or the user calls `fExecuteGraph`. */
void fDisableEagerExecution();

/** Returns 1 if eager execution has been enabled, else 0 */
int fIsEagerExecution();

/** The 4 allowed data types:
 * - `F_INT32`(integer, 32bit)
 * - `F_INT64`(integer, 64bit)
 * - `F_FLOAT32` (floating point, 32bit)
 * - `F_FLOAT64` (floating point, 64bit)
 */
enum FType { F_INT32, F_INT64, F_FLOAT32, F_FLOAT64 };

enum FOperationType {
	FSTORE,
	FGEN_RANDOM,
	FGEN_CONSTANT,
	FGEN_ARANGE,
	FADD,
	FSUB,
	FMUL,
	FDIV,
	FPOW,
	FNEG,
	FLOG,
	FSIGN,
	FEVEN,
	FLOG2,
	FLOG10,
	FSIN,
	FCOS,
	FTAN,
	FASIN,
	FACOS,
	FATAN,
	FSQRT,
	FEXP,
	FLATTEN,
	FMATMUL,
	FCONVERSION,
	FRESHAPE,
	FMIN,
	FMAX,
	FREDUCE_SUM,
	FREDUCE_MUL,
	FREDUCE_MIN,
	FREDUCE_MAX,
	FSLICE,
	FABS,
	FREPEAT,
	FTRANSPOSE,
	FEXTEND,
	FCONCAT,
	FLESS,
	FEQUAL,
	FGREATER,
	FCONVOLVE,
	FGRADIENT_CONVOLVE1, // only for internal use!
	FGRADIENT_CONVOLVE2, // only for internal use!
	FINDEX,
	FSET_INDEX,
	FSLIDING_WINDOW,
	FUNSLIDE_WINDOW,
	FPOOLING_MAX,
	FPOOLING_SUM,
	FGRADIENT_POOLING_MAX, // only for internal use
	FDROPOUT,
	FNUM_OPERATION_TYPES
};

/**
 * Describes one operation. An operation always has a shape, described by
 * `FOperation.shape` which is an array of size
 * `FOperation.dimensions` with each entry denoting the size of the
 * corresponding dimension. `FOperation.op_type` denotes the type of
 * operation, `FOperation.data_type` the type of the underlying data,
 * `FOperation.additional_data` is operation specific.*/
struct FOperation {
		size_t *shape;
		void *additional_data;
		// type of operation, to enable switch cases and avoid v-table lookups
		enum FOperationType op_type;
		// datatype of result
		enum FType data_type;
		// shape of the data after execution
		int dimensions;
		// currently a boolean indicating if standard broadcasting (0) is to be
		// used or inverse (1), in the future maybe an additional indicators for
		// more advanced broadcasting methods may be implemented
		int broadcasting_mode;
};
typedef struct FOperation FOperation;

/** Stores the resulting data after an execution of `fExecuteGraph` (or implicit
 * execution). The data can be found in `FResultData.data`, the datatype in
 * `FOperation.data_type` of the corresponding `FGraphNode`.
 * The number of entries (not number of bytes) is stored in
 * `FResultData.num_entries`. The data may be consistently modified
 * if...
 * - ...the data size is changed, num_entries is equivalently updated and
 *      `realloc` is used and ...
 * - ...the data was not already loaded to the gpu (i.e. the result must be the
 *        return value of `fExecuteGraph_cpu`)
 */
struct FResultData {
		// link to gpu data
		cl_mem mem_id;
		void *data;
		size_t num_entries;
};
typedef struct FResultData FResultData;

/** Describes one node in the Graph. Stores the corresponding operation in
 * `FGraphNode.operation`, an array of predecessors (the arguments of
 * the operation) in `FGraphNode.predecessors`, its size in
 * `FGraphNode.num_predecessor` and the reference counter in
 * `FGraphNode.reference_counter`. Do not modify any parameter by yourself,
 * since the framework manages them, but you can read the data and structure
 * from them. The nodes are allocated by the operation functions, they and their
 * members should neither be manually created, edited or freed except by the
 * corresponding flint methods. */
struct FGraphNode {
		int num_predecessor;
		struct FGraphNode **predecessors;
		FOperation operation; // the operation represented by this graph node
		size_t reference_counter; // for garbage collection in free graph
		FResultData *result_data; // to store computational result
		void *gradient_data; // to store a list of present variables that are
							 // currently watched in the graph
};
typedef struct FGraphNode FGraphNode;

/** Result of an call to `fCreateGraph`, see `FResultData`.
 * Data of this Operation may not be changed manually when using a GPU Backend.
 */
struct FStore {
		// link to gpu data
		cl_mem mem_id;
		void *data;
		size_t num_entries;
};
// range instructions
struct FSlice {
		long *start;
		long *end;
		long *step;
};
struct FExtend {
		size_t *start;
		long *step;
};
struct FSlidingWindow {
		size_t *size;
		unsigned int *step;
};

// functions

/**
 * - `data`: pointer to the flattened data array that should be loaded into
 *   the node
 * - `num_entries`: the number of elements (NOT BYTES!) that should
 *   be loaded
 * - `data_type`: the datatype of `data`
 * - `shape`: an array of size `dimensions`, each entry describing the
 *   size of the corresponding dimension. Make sure, `data` is at least as
 *   large as the product of all entries in `shape`
 * - `dimensions`: the number of dimensions
 *
 * Creates a Graph with a single store instruction, the data is
 * copied to intern memory, so after return of the function, `data` and `shape`
 * may be deleted. */
FGraphNode *fCreateGraph(const void *data, const int num_entries,
						 const enum FType data_type, const size_t *shape,
						 const int dimensions);

/** Creates a tensor that contains the single given values in all entries
 *
 * - `value`: the value this tensor should consist of
 * - `shape`: an array of size `dimensions`, each entry describing the size
 *    of the corresponding dimension.
 * - `dimensions`: the number of dimensions
 */
FGraphNode *fconstant_i(const int value, const size_t *shape,
						const int dimensions);

/** Creates a tensor that contains the single given values in all entries
 *
 * - `value`: the value this tensor should consist of
 * - `shape`: an array of size `dimensions`, each entry describing the size
 *    of the corresponding dimension.
 * - `dimensions`: the number of dimensions
 */
FGraphNode *fconstant_l(const long value, const size_t *shape,
						const int dimensions);

/** Creates a tensor that contains the single given values in all entries
 *
 * - `value`: the value this tensor should consist of
 * - `shape`: an array of size `dimensions`, each entry describing the size
 *    of the corresponding dimension.
 * - `dimensions`: the number of dimensions
 */
FGraphNode *fconstant_f(const float value, const size_t *shape,
						const int dimensions);

/** Creates a tensor that contains the single given values in all entries
 *
 * - `value`: the value this tensor should consist of
 * - `shape`: an array of size `dimensions`, each entry describing the size
 *    of the corresponding dimension.
 * - `dimensions`: the number of dimensions
 */
FGraphNode *fconstant_d(const double value, const size_t *shape,
						const int dimensions);

/** Creates a tensor that contains randomly distributed values in the range of
 * [0, 1)
 *
 * - `shape`: an array of size `dimensions`, each entry describing the size
 *    of the corresponding dimension.
 * - `dimensions`: the number of dimensions
 */
FGraphNode *frandom(const size_t *shape, const int dimensions);

/** Creates a int64 tensor that contains the indices relative to a given
 * dimension `ax` for each element, i.e. each entry is its index in that
 * corresponding dimension. If you need to index more than one dimension, create
 * multiple such tensors with `arange`.
 */
FGraphNode *farange(const size_t *shape, const int dimensions, const int ax);

/** Decrements `FGraphNode.reference_counter` of `graph` (for reference
 * counting) and deallocates the node and its corresponding data, if the counter
 * reaches 0. If the node is deallocated, the same process is repeated with its
 * predecessors. So you can safely connect nodes multiple times and have only to
 * free the leaf nodes (i.e. the results), without caring about cross-reference,
 * since those are handled by the reference counting system.*/
void fFreeGraph(FGraphNode *graph);

/** Executes the graph node operations from all yet to be executed predecessors
 * to `node` and returns a node with a `FResultData` operation in
 * which the resulting data is stored.
 *
 * If the graph is executed by the GPU
 * backend, a opencl kernel containing all selected operations (the nodes
 * operation and those indirect parent operations which were not yet
 * executed) are compiled and executed. The kernels are cached, so it improves
 * the performance of a program if the same graph-structures are reused (not
 * necessary the same nodes, but the same combination of nodes), since then the
 * backend can reuse already compiled kernels. The GPU backend is allowed to
 * keep the Result data on the GPU without synchronizing it (so the generated
 * `FResultData` may not have a `data` member!). To force synchronization use
 * `fSyncMemory` or replace the call with `fCalculateResult`.
 *
 * If the CPU backend is chosen, it
 * does not matter, since every operation is executed independently (here eager
 * execution might be faster).
 *
 * The backend is selected by the framework if both
 * are initialized, else the one that is initialized will be chosen. If both are
 * uninitialized, both will be initialized prior to execution. If eager
 * execution is enabled each node will be executed eagerly upon construction or
 * with this method.
 *
 * Also see `fEnableEagerExecution`, `fSyncMemory`*/
FGraphNode *fExecuteGraph(FGraphNode *node);

/** Executes the graph node operations from all yet to be executed predecessors
 * to `node` and returns a node with a `FResultData` operation in
 * which the resulting data is stored. */
FGraphNode *fExecuteGraph_cpu(FGraphNode *node);

/** Executes the graph node operations from all yet to be executed predecessors
 * to `node` and returns a node with a `FResultData` operation in
 * which the resulting data is stored. For the GPU
 * backend, a opencl kernel containing all selected operations (the nodes
 * operation and those indirect parent operations which were not yet executed)
 * are compiled and executed. The kernels are cashed, so it improves the
 * performance of a program if the same graph-structures are reused (not
 * necessary the same nodes, but the same combination of nodes), since then the
 * backend can reuse already compiled kernels. */
FGraphNode *fExecuteGraph_gpu(FGraphNode *node);

/** Executes the graph node directly and assumes that predecessor data has
 * already been computed. Uses the CPU backend. Mainly used by helper functions
 * of the framework, only use it if you now what you are doing.
 *
 * Also see `fEnableEagerExecution`, `fExecuteGraph_cpu` and `fExecuteGraph`*/
FGraphNode *fExecuteGraph_cpu_eagerly(FGraphNode *node);

/** Executes the graph node directly and assumes that predecessor data has
 * already been computed. Uses the GPU backend where for each operation and
 * parameter type combination one eager kernel will be computed. Mainly used by
 * helper functions of the framework, only use it if you now what you are doing.
 *
 *  Also see `fEnableEagerExecution`, `fExecuteGraph_gpu` and `fExecuteGraph`*/
FGraphNode *fExecuteGraph_gpu_eagerly(FGraphNode *node);

/** `fExecuteGraoh` does not guarantee that memory is present on the cpu (it may
 * be kept on the GPU for performance reasons). This method enforces all GPU
 * data to be flushed to the CPU (but never executes the node!).
 *
 * Also see `fCalculateResult`
 */
FResultData *fSyncMemory(FGraphNode *data);

/**
 * Convenience Method that first execute `fExecuteGraph` and then `fSyncMemory`
 * on the node.
 *
 * It represents execution with one of both backends and additional memory
 * synchronizing for the gpu framework.
 */
FGraphNode *fCalculateResult(FGraphNode *node);

//  gradient calculation

/** Calculates the overall gradient of an output node to a variable.
 * The variable must be marked as a gradient variable, see
 * `fMarkGradientVariable` and the output node `outputfct` must be constructed
 * in an gradient context (to remember which variables are directly or
 * indirectly present in which operation), which can be started with
 * `fStartGradientContext`. If you need to compute multiple gradients for one
 * output use `fCalculateGradients` since it is far more efficient.
 *
 * - `outputfct`: the Node which represents the chain of functions of which
 *    the gradient is to be computed.
 * - `dx`: the variable for which outputfct is derived for
 */
FGraphNode *fCalculateGradient(FGraphNode *outputfct, FGraphNode *dx);

/** Calculates the overall gradient of an output node to multiple variables.
 * The variables must be marked as a gradient variable, see
 * `fMarkGradientVariable` and the output node `outputfct` must be constructed
 * in an gradient context (to remember which variables are directly or
 * indirectly present in which operation), which can be started with
 * `fStartGradientContext`.
 *
 * - `outputfct`: the Node which represents the chain of functions of which
 *    the gradients are to be computed.
 * - `dx`: array of variables for which outputfct is derived for.
 * - `num_gradients`: the number of variables in `dx`.
 * - `gradients`: an array of size `num_gradients` in which the resulting
 *    gradients will be stored per variable.
 */
FErrorType fCalculateGradients(FGraphNode *outputfct, FGraphNode **dx,
							   const unsigned int num_gradients,
							   FGraphNode **gradients);

/** Starts a gradient context, gradient information will be inherited until the
 * next call to `fStopGradientContext`. A history containing information about
 * all watched nodes in the parent graph is kept up to date within a gradient
 * context for each node created in it. The node does not have to be watched in
 * that particular context (it just has to be marked as watched), but only for
 * operations which are constructed inside such a gradient context a gradient to
 * a variable may be computed. Since this context introduces overhead and limits
 * `fOptimizeMemory` significantly, it should be closed as soon as possible. */
void fStartGradientContext();

/** Stops a gradient context, all inherited gradient information and watched
 * nodes will be kept, but no longer inherited to new ones. */
void fStopGradientContext();

/** Return true if the call to this function is placed within a gradient
 * context. */
bool fIsGradientContext();

/** Marks this node as a node for which a gradient might be calculated later.
 * It is only possible to calculate the gradient for this node (as a derivative)
 * in operations that occur AFTER a call to this method (all subsequent
 * operations will have a remark that enlists this node as a possible gradient
 * variable, to enable less memory usage and faster gradient calculation).
 */
void fMarkGradientVariable(FGraphNode *node);

/** Removes the gradient mark (and subsequent memory overhead) for this node.
 * After a call to this method no subsequent gradient calculations with this
 * node as a derivative will be possible.
 */
void fUnmarkGradientVariable(FGraphNode *node);

/** Optimizes memory by freeing all parental data (operand nodes of the
 * operation of this node) and transforming this node to a storage nodes
 * if no gradient variables are present in this node and result data is
 * present (i.e. it has been executed).
 * If you call this function manually please make sure to increase the
 * `reference_counter` of the parents if you want to keep their handles,
 * since this function may decrease their counter and free them.
 * The C++ framework does this automatically.
 */
FGraphNode *fOptimizeMemory(FGraphNode *node);

/** Sometimes there are combinations of nodes where both normal and inverse
 * broadcasting is possible, but yields different results, e.g. multiplication
 * for two nodes with shapes [3, 5, 3, 5] and [3, 5]. The framework chooses
 * normal broadcasting over inverse if both are possible, this function allows
 * you to alter this behaviour and mark a node to be inversely broadcasted.
 * After the call to this function the given node will from then on only
 * inversely broadcasted (in cases where only normal broadcasting is available
 * an error will occur!). It has no effect in operations that don't use
 * broadcasting. You can "unmark" the node with `fUnenforceInverseBroadcasting`.
 */
void fEnforceInverseBroadcasting(FGraphNode *node);

/** Undos `fEnforceInverseBroadcasting` for a node.*/
void fUnenforceInverseBroadcasting(FGraphNode *node);

//  operations

/** Serializes the data and shape of the node and returns an array of chars in
 * which the serialized data will be written (binary data, not a string). The
 * returned array is allocated on the systems heap with `malloc`, so you have to
 * free it after you are done with it. The number of bytes that the returned
 * array has is written into the memory `bytes_written` points to if it is not a
 * nullptr. If the node doesn't have result data, it is executed first.
 */
char *fserialize(FGraphNode *node, size_t *bytes_written);

/** Unserializes data generated by `fserialize`.
 * The size of the node is stored in the data and the complete node is read.
 * The number of bytes read is stored in `bytes_read`. Internally calls
 * `fCreateGraph`.
 */
FGraphNode *fdeserialize(char *data, size_t *bytes_read);

/** Loads an image from the given path.
 * The image will be stored in floating point data and the shape will be h, w, c
 * where w is the width, h is the height and c are the channels.
 * Supported formats include png, jpeg, bmp, gif, hdr ... essentially everything
 * stb_image supports
 */
FGraphNode *fload_image(const char *path);

FErrorType fstore_image(FGraphNode *node, const char *path,
						enum FImageFormat format);

/** Elementwise addition of `a` and `b`, i.e. `a[i] + b[i]`. */
FGraphNode *fadd_g(FGraphNode *a, FGraphNode *b);
/** Elementwise substraction of `a` and `b`, i.e. `a[i] - b[i]`. */
FGraphNode *fsub_g(FGraphNode *a, FGraphNode *b);
/** Elementwise division of `a` and `b`, i.e. `a[i] / b[i]`. */
FGraphNode *fdiv_g(FGraphNode *a, FGraphNode *b);
/** Elementwise multiplication of `a` and `b`, i.e. `a[i] * b[i]`. */
FGraphNode *fmul_g(FGraphNode *a, FGraphNode *b);
/** Elementwise power of `a` and `b`, i.e. `pow(a[i], b[i])`. */
FGraphNode *fpow_g(FGraphNode *a, FGraphNode *b);

/** Elementwise addition of a and b, i.e. `a[i] + b`. */
FGraphNode *fadd_ci(FGraphNode *a, const int b);
/** Elementwise addition of a and b, i.e. `a[i] + b`. */
FGraphNode *fadd_cl(FGraphNode *a, const long b);
/** Elementwise addition of a and b, i.e. `a[i] + b`. */
FGraphNode *fadd_cf(FGraphNode *a, const float b);
/** Elementwise addition of a and b, i.e. `a[i] + b`. */
FGraphNode *fadd_cd(FGraphNode *a, const double b);

/** Elementwise subtraction of a and b, i.e. `a[i] - b`. */
FGraphNode *fsub_ci(FGraphNode *a, const int b);
/** Elementwise subtraction of a and b, i.e. `a[i] - b`. */
FGraphNode *fsub_cl(FGraphNode *a, const long b);
/** Elementwise subtraction of a and b, i.e. `a[i] - b`. */
FGraphNode *fsub_cf(FGraphNode *a, const float b);
/** Elementwise subtraction of a and b, i.e. `a[i] - b`. */
FGraphNode *fsub_cd(FGraphNode *a, const double b);

/** Elementwise subtraction of a and b, i.e. `a - b[i]`. */
FGraphNode *fsub_ici(const int a, FGraphNode *b);
/** Elementwise subtraction of a and b, i.e. `a - b[i]`. */
FGraphNode *fsub_icl(const long a, FGraphNode *b);
/** Elementwise subtraction of a and b, i.e. `a - b[i]`. */
FGraphNode *fsub_icf(const float a, FGraphNode *b);
/** Elementwise subtraction of a and b, i.e. `a - b[i]`. */
FGraphNode *fsub_icd(const double a, FGraphNode *b);

/** Elementwise division of a and b, i.e. `a[i] / b`. */
FGraphNode *fdiv_ci(FGraphNode *a, const int b);
/** Elementwise division of a and b, i.e. `a[i] / b`. */
FGraphNode *fdiv_cl(FGraphNode *a, const long b);
/** Elementwise division of a and b, i.e. `a[i] / b`. */
FGraphNode *fdiv_cf(FGraphNode *a, const float b);
/** Elementwise division of a and b, i.e. `a[i] / b`. */
FGraphNode *fdiv_cd(FGraphNode *a, const double b);

/** Elementwise division of a and b, i.e. `a / b[i]`. */
FGraphNode *fdiv_ici(const int a, FGraphNode *b);
/** Elementwise division of a and b, i.e. `a / b[i]`. */
FGraphNode *fdiv_icl(const long a, FGraphNode *b);
/** Elementwise division of a and b, i.e. `a / b[i]`. */
FGraphNode *fdiv_icf(const float a, FGraphNode *b);
/** Elementwise division of a and b, i.e. `a / b[i]`. */
FGraphNode *fdiv_icd(const double a, FGraphNode *b);

/** Elementwise multiplication of a and b, i.e. `a[i] * b`. */
FGraphNode *fmul_ci(FGraphNode *a, const int b);
/** Elementwise multiplication of a and b, i.e. `a[i] * b`. */
FGraphNode *fmul_cl(FGraphNode *a, const long b);
/** Elementwise multiplication of a and b, i.e. `a[i] * b`. */
FGraphNode *fmul_cf(FGraphNode *a, const float b);
/** Elementwise multiplication of a and b, i.e. `a[i] * b`. */
FGraphNode *fmul_cd(FGraphNode *a, const double b);

/** Takes the elementwise power of a to b, i.e. `pow(a[i], b)`.*/
FGraphNode *fpow_ci(FGraphNode *a, const int b);
/** Takes the elementwise power of a to b, i.e. `pow(a[i], b)`.*/
FGraphNode *fpow_cl(FGraphNode *a, const long b);
/** Takes the elementwise power of a to b, i.e. `pow(a[i], b)`.*/
FGraphNode *fpow_cf(FGraphNode *a, const float b);
/** Takes the elementwise power of a to b, i.e. `pow(a[i], b)`.*/
FGraphNode *fpow_cd(FGraphNode *a, const double b);

/** Takes the elementwise natural logarithm of `a` */
FGraphNode *flog(FGraphNode *a);
/** Takes the elementwise logarithm of `a` to the basis of 2 */
FGraphNode *flog2(FGraphNode *a);
/** Takes the elementwise logarithm of `a` to the basis of 10 */
FGraphNode *flog10(FGraphNode *a);
/** Takes the elementwise sinus of `a` */
FGraphNode *fsin(FGraphNode *a);
/** Takes the elementwise square root of `a` */
FGraphNode *fsqrt_g(FGraphNode *a);
/** Takes the elementwise exponential of `a` (power of the constant `e` to `a`)
 */
FGraphNode *fexp(FGraphNode *a);
/** Takes the elementwise cosinus of `a` */
FGraphNode *fcos(FGraphNode *a);
/** Takes the elementwise tangents of `a` */
FGraphNode *ftan(FGraphNode *a);
/** Takes the elementwise inverse sinus of `a` */
FGraphNode *fasin(FGraphNode *a);
/** Takes the elementwise inverse cosinus of `a` */
FGraphNode *facos(FGraphNode *a);
/** Takes the elementwise inverse tangents of `a` */
FGraphNode *fatan(FGraphNode *a);
/** Negates the elements of `a`, i.e. `-a[i]` */
FGraphNode *fneg(FGraphNode *a);
/** Returns a `F_INT32` tensor x with the shape of a with `x[i] = 1` if `a[i] >=
 * 0` else `x[i] = -1`. `a` needs to have a integer data type.*/
FGraphNode *fsign(FGraphNode *a);
/** Returns a `F_INT32` tensor `x` with the shape of `a` with `x[i] = 1` if
 * `a[i] % 2 = 0` else `x[i] = 0`.
 */
FGraphNode *feven(FGraphNode *a);
/** Compares two tensors elementwise by `a < b` and returns a 0,1 `F_INT32`
 * Tensor. `0` denotes that `a >= b`, `1` that `a < b`. */
FGraphNode *fless_g(FGraphNode *a, FGraphNode *b);
/** Compares two tensors elementwise by `a > b` and returns a 0,1 `F_INT32`
 * Tensor. */
FGraphNode *fgreater_g(FGraphNode *a, FGraphNode *b);
/** Compares two tensors elementwise by `a = b`` and returns a 0,1 `F_INT32`
 * Tensor. `0` denotes that `a <= b`, `1` that `a > b`.*/
FGraphNode *fequal_g(FGraphNode *a, FGraphNode *b);
/** Compares a tensor and a constant elementwise by `a < b` and returns a 0,1
 * `INT32` Tensor. See `fless_g`. */
FGraphNode *fless_ci(FGraphNode *a, const int b);
/** Compares a tensor and a constant elementwise by `a < b` and returns a 0,1
 * `INT32` Tensor. See `fless_g`. */
FGraphNode *fless_cl(FGraphNode *a, const long b);
/** Compares a tensor and a constant elementwise by `a < b` and returns a 0,1
 * `INT32` Tensor. See `fless_g`. */
FGraphNode *fless_cf(FGraphNode *a, const float b);
/** Compares a tensor and a constant elementwise by `a < b` and returns a 0,1
 * `INT32` Tensor. See `fless_g`. */
FGraphNode *fless_cd(FGraphNode *a, const double b);
/** Compares a tensor and a constant elementwise by `a > b` and returns a 0,1
 * `INT32` Tensor. See `fgreater_g`. */
FGraphNode *fgreater_ci(FGraphNode *a, const int b);
/** Compares a tensor and a constant elementwise by `a > b` and returns a 0,1
 * `INT32` Tensor. See `fgreater_g`. */
FGraphNode *fgreater_cl(FGraphNode *a, const long b);
/** Compares a tensor and a constant elementwise by `a > b` and returns a 0,1
 * `INT32` Tensor. See `fgreater_g`. */
FGraphNode *fgreater_cf(FGraphNode *a, const float b);
/** Compares a tensor and a constant elementwise by `a > b` and returns a 0,1
 * `INT32` Tensor. See `fgreater_g`. */
FGraphNode *fgreater_cd(FGraphNode *a, const double b);
/** Compares a tensor and a constant elementwise by `a = b` and returns a 0,1
 * `INT32` Tensor. See `fequal_g`. */
FGraphNode *fequal_ci(FGraphNode *a, const int b);
/** Compares a tensor and a constant elementwise by `a = b` and returns a 0,1
 * `INT32` Tensor. See `fequal_g`. */
FGraphNode *fequal_cl(FGraphNode *a, const long b);
/** Compares a tensor and a constant elementwise by `a = b` and returns a 0,1
 * `INT32` Tensor. See `fequal_g`. */
FGraphNode *fequal_cf(FGraphNode *a, const float b);
/** Compares a tensor and a constant elementwise by `a = b` and returns a 0,1
 * `INT32` Tensor. See `fequal_g`. */
FGraphNode *fequal_cd(FGraphNode *a, const double b);
/** Carries out matrix multiplication on the last two dimensions of the tensors.

 * E.g. a matrix multiplication of two tensors with shapes `(64, 32, 16)` and
 * `(16, 24)` will yield a tensor with shape `(64, 32, 24)`.
 *
 * Since for one entry of the
 * tensor multiple other previous entries are needed, the operand tensors need
 * to be executed first. Therefor the method will implicitly (or eagerly)
 * execute the two parameter nodes `a` and `b` if their data is not already
 * present. */
FGraphNode *fmatmul(FGraphNode *a, FGraphNode *b);
/** Flattens the complete tensor to a tensor with one
dimension.
E.g.`flattened([[[3, 1, 4], [2, 1, 5]], [[0, 4, 2], [4, 7, 9]]]) = [3, 1, 4, 2,
1, 5, 0, 4, 2, 4, 7, 9]`.
*/
FGraphNode *fflatten(FGraphNode *a);
/** Flattens a tensor `a` with `n` dimensions along
`dimension`, resulting in a tensor with `n-1` dimensions.
Flattening a dimension will remove it from the shape of the tensor, therefor
it's not possible to flatten the dimension 0. A Tensor `[[[3, 1, 4], [2, 1, 5]],
[[0, 4, 2], [4, 7, 9]]]` flattened along dimension 1 will result in `[[3,1,4],
[2,1,5], [0,4,2], [4,7,9]]`.
*/
FGraphNode *fflatten_dimension(FGraphNode *a, int dimension);

/** Converts the data of `a` to the type given by `newtype`*/
FGraphNode *fconvert(FGraphNode *a, enum FType newtype);
/** Reshapes the underlying data of the tensor to the new shape. The product of
  each dimension of the new shape must be the same as the product of the
  dimensions of the previous shape (i.e. it must describe the same number of
  entries of the tensor).*/
FGraphNode *freshape(FGraphNode *a, const size_t *newshape,
					 const int dimensions);
/** Takes the minimum of two tensors element wise along the last dimension of
 * each, i.e. `a[i]` if `a[i] < b[i]` else `b[i]` */
FGraphNode *fmin_g(FGraphNode *a, FGraphNode *b);
/** Takes the minimum of two tensors element wise along the last dimension of
 * each, i.e. `a[i]` if `a[i] < b` else `b` */
FGraphNode *fmin_ci(FGraphNode *a, const int b);
/** Takes the minimum of two tensors element wise along the last dimension of
 * each, i.e. `a[i]` if `a[i] < b` else `b` */
FGraphNode *fmin_cl(FGraphNode *a, const long b);
/** Takes the minimum of two tensors element wise along the last dimension of
 * each, i.e. `a[i]` if `a[i] < b` else `b` */
FGraphNode *fmin_cf(FGraphNode *a, const float b);
/** Takes the minimum of two tensors element wise along the last dimension of
 * each, i.e. `a[i]` if `a[i] < b` else `b` */
FGraphNode *fmin_cd(FGraphNode *a, const double b);
/** Takes the maximum of two tensors element wise along the last dimension of
 * each, i.e. `a[i]` if `a[i] > b[i]` else `b[i]` */
FGraphNode *fmax_g(FGraphNode *a, FGraphNode *b);
/** Takes the maximum of two tensors element wise along the last dimension of
 * each, i.e. `a[i]` if `a[i] > b` else `b` */
FGraphNode *fmax_ci(FGraphNode *a, const int b);
/** Takes the maximum of two tensors element wise along the last dimension of
 * each, i.e. `a[i]` if `a[i] > b` else `b` */
FGraphNode *fmax_cl(FGraphNode *a, const long b);
/** Takes the maximum of two tensors element wise along the last dimension of
 * each, i.e. `a[i]` if `a[i] > b` else `b` */
FGraphNode *fmax_cf(FGraphNode *a, const float b);
/** Takes the maximum of two tensors element wise along the last dimension of
 * each, i.e. `a[i]` if `a[i] > b` else `b` */
FGraphNode *fmax_cd(FGraphNode *a, const double b);
/** Reduces one dimension of the tensor by additive folding e.g.
 *
 * `freduce_sum([[1,2,3], [4,5,6]], 0) = [5,7,9]`,
 * `freduce_sum([[1,2,3], [4,5,6]], 1) = [6,15]`
 *
 * The results of the predecessor node must be available, to
 * ensure that the method may execute the parameter node. */
FGraphNode *freduce_sum(FGraphNode *a, const int dimension);

/** Reduces one dimension of the tensor by multiplicative folding e.g.
 *
 * `freduce_mul([[1,2,3], [4,5,6]], 0) = [4,10,18]`,
 * `freduce_mul([[1,2,3], [4,5,6]], 1) = [6, 120]`
 *
 * The results of the predecessor node must be available; to
 * ensure that the method may execute the parameter node.*/
FGraphNode *freduce_mul(FGraphNode *a, const int dimension);

/** Reduces one dimension of the tensor by keeping the minimum e.g.
 *
 * `freduce_min([[1,32,3], [4,5,3]], 0) = [1,5,3]`,
 * `freduce_min([[9,2,3], [-1,5,6]], 1) = [2, -1]`
 *
 * The results of the predecessor node must be available; to
 * ensure that the method may execute the parameter node.*/
FGraphNode *freduce_min(FGraphNode *a, const int dimension);

/** Reduces one dimension of the tensor by keeping the maximum e.g.
 *
 * `freduce_max([[1,32,3], [4,5,3]], 0) = [4,32,3]`,
 * `freduce_max([[9,2,3], [-1,5,6]], 1) = [9, 6]`
 *
 * The results of the predecessor node must be available; to
 * ensure that the method may execute the parameter node.*/
FGraphNode *freduce_max(FGraphNode *a, const int dimension);

/** Selects a slice of the tensor with a dimension wise start and end index.
 * `start` and `end` are arrays with as many entries
 * as the tensor has dimensions. They may contain negative values,
 * which are then subtracted from the end of the tensor (e.g. `-1` means the
 * element before the last element). `start` is inclusive and describes the
 * start index of the selection per dimension and `end` describes the end index
 * per dimension and is exclusive.
 */
FGraphNode *fslice(FGraphNode *a, const long *start, const long *end);

/** Selects a slice of the tensor with a dimension wise start index, end index
 * and step size. `start`, `end` and `step` are arrays with as
 * many entries as the tensor has dimensions. `start` and `end` may
 * contain negative values, which are then subtracted from the end of the tensor
 * (e.g. `-1` means the element before the last element). `start` is inclusive
 * and describes the start index of the selection per dimension and `end`
 * describes the end index per dimension and is exclusive. `step` contains the
 * per dimension step size (e.g. `2` meaning every second element will be
 * selected etc.) and may be negative as well, which reverses the traversal
 * order (the first elements are selected as the last ones). For a negative step
 * size, `start > end` must hold (for a positive of course `end > start`) for
 * each dimension.
 */
FGraphNode *fslice_step(FGraphNode *a, const long *start, const long *end,
						const long *step);

/**
 * Creates a new tensor of zeroes with the requested shape. The original tensor
 * is embedded at the given indices.
 * - `a` original tensor which shape is to be extended
 * - `new_shape` array of new sizes per dimension. Has `dimension` number of
 *    entries
 * - `insert_at` array with indices per dimension denoting where `a` is to be
 *    placed in the resulting tensor
 */
FGraphNode *fextend(FGraphNode *a, const size_t *new_shape,
					const size_t *insert_at);

/**
 * Creates a new tensor of zeroes with the requested shape. The original tensor
 * is embedded at the given indices.
 * - `a` original tensor which shape is to be extended,
 * - `new_shape` array of new sizes per dimension. Has `dimension` number of
 *    entries.
 * - `insert_at` array with indices per dimension denoting where `a` is to be
 *    placed in the resulting tensor. Has a value per dimension.
 * - `step_size` allows to pull apart `a`, emplacing `step_size[i]` 0 between
 *    each value of `a`. Has a value per dimension.
 */
FGraphNode *fextend_step(FGraphNode *a, const size_t *new_shape,
						 const size_t *insert_at, const long *step_size);

/**
 * Concats two nodes with each other along an axis.
 * The nodes have to have the same type and dimensions.
 * e.g. `fconcat({[[0, 1], [2, 3]], [[4, 5], [6, 7]]}, 0) = [[0, 1], [2, 3], [4,
 * 5], [6, 7]]`
 *
 * `fconcat({[[0, 1], [2, 3]], [[4, 5], [6, 7]]}, 1) = [[0, 1, 4, 5], [2, 3, 6,
 * 7]]`
 */
FGraphNode *fconcat(FGraphNode *a, FGraphNode *b, const unsigned int axis);

/**
 * Adds a new dimension at an arbitrary position to the tensor and repeats the
 * following dimensions to match a given shape.
 *
 * - `ax` the dimension prior to which the new dimension will be inserted (`0`
 *    means a new dimension in the front, `n` means as a new last
 *    dimension).
 * - `ax_size` the new size of that dimension (repeats the following
 *    dimensions `ax_size - 1` times).
 */
FGraphNode *fexpand(FGraphNode *a, const unsigned int axis,
					const unsigned int size);

/** Takes the elementwise absolute value of `a`, i.e. `|a[i]|` */
FGraphNode *fabs_g(FGraphNode *a);

/** Repeats dimensions of a tensor multiple times
 *
 * - `a`: the node in which dimensions are to be repeated
 * - `repititions`: array with the same number of entries as the tensor has
 *    dimensions
 * e.g. `repeat([[0,1], [2,3]], [2, 3]) = [[0,1,0,1,0,1],
 * [2,3,2,3,2,3], [0,1,0,1,0,1], [2,3,2,3,2,3]]`
 */
FGraphNode *frepeat(FGraphNode *a, int *repititions);

/** Transposes this tensor along multiple dimensions
 *
 * - `a`: the node which should be transposed
 * - `transpositions`: an array with the same number of entries as the tensor
 *   has dimensions, which gives the perumtation of dimensions.
 * The tensor will have a resulting shape in which the size in dimension `i`
 * corresponds to the former size in dimension `transpositions[i]`.
 */
FGraphNode *ftranspose(FGraphNode *a, int *transpositions);

/** Convolves the `n`-dimensional input tensor `a` with a `n` or
 * `n+1`-dimensional filter kernel `kernel` and a per dimensional step size
 * `steps` with size of `n-1`. It is expected that `a` and `kernel` have the
 * same size in their last dimension (which will be completely reduced by the
 * convolution). In all other dimensions the size of `a` should be larger or
 * equal to the size of `kernel` or - if kernel has a dimensionality of `n+1` -
 * every `i`th dimension of `a` should be learger or equal to the dimension `i +
 * 1` of kernel, since the first dimension contains each filter. The `kernel`
 * will be 'slid' over `a` in each dimension, multiplying all values of `kernel`
 * with the corresponding ones in `a` and summing them up to a single value and
 * moving the kernel further by the value given in `steps` in that corresponding
 * dimension.
 *
 * If the filter is `n+1`-dimensional the first dimension of `kernel` is viewed
 * as an array of filters in which for each a normal convolution is executed and
 * each filter result is added to a new last dimension, meaning the result will
 * instead have `n`-dimensionality where the reduced last dimension is replaced
 * by the result of each filter convolution (normal semantic for a convolution
 * with `k` filters in machine learning. Here the semantic representation of the
 * shape of `kernel` would be `(k, 1, kernel_size, ...)` where `1` is inserted
 * for the batch dimension).
 *
 * The implementation does not include any padding, meaning only convolutions
 * where the complete kernel still fits into the array will be executed (the
 * shape will be calculated correspondingly). If you want to modify this
 * behaviour (i.e. include padding) you can use `extend`, `slice` or similar.
 *
 * The resulting Tensor will therefor have a shape with dimensionality `n - 1`
 * if kernel is also `n` dimensional and size of `(shape[i] -
 * kernel.get_shape()[i] - 1) / steps[i]` if `(shape[i] - kernel.get_shape()[i]
 * - 1)` is divisable by `steps[i]` else `(shape[i] - kernel.get_shape()[i] - 1)
 * / steps[i] + 1`. If the kernel is `n+1` dimensional the result has
 * dimensionality of `n` with the same shape, except for the last dimension
 * where it will have the numbers of filters (the first dimension of kernel) as
 * its size.
 */
FGraphNode *fconvolve(FGraphNode *a, FGraphNode *kernel,
					  const unsigned int *steps);

/**
 * Selects single elements with a index-tensor (integer tensor containing
 * indices for the selected dimension).
 * It indexes a dimension of the input tensor and the result has
 * the shape of the input tensor except for the indexed dimension.
 * It is assumed that except for the last entry its shape is a prefix of the
 * shape of the input tensor and the indexing will occur in the matched subsets
 * (the last dimension of the `indices` Tensor is the one indexed in `a`).
 * E.g.
 *
 * `findex([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [1, 0])
 * = [[[4, 5], [6, 7]], [[0, 1], [2, 3]]]`
 *
 * `findex([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [0, 0, 1])
 *  = [[[0, 1], [0, 1], [1, 2]], [[3, 4], [3, 4], [5, 6]], [[7, 8], [7, 8],
 * [9, 10]]]`
 *
 * `findex([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[0], [1],
 * [0]]) = [[[0], [2]], [[5], [7]], [[8], [10]]]`
 *
 * `findex([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[0, 0],
 * [1, 0], [0, 1]]) = [[[0, 0], [2, 2]], [[5, 4], [7, 6]], [[8, 9], [10,
 * 11]]]`
 */
FGraphNode *findex(FGraphNode *a, FGraphNode *indices);

/**
 * Assigns to each element in `b` one element in `a` where that element will be
 * "send" to, i.e. the place in `a` the index points to will be set to the
 * corresponding element from `b`. If multiple elements from `b` are sent to the
 * same place in `a` they will be summed up. The shape of `indices` must be a
 * prefix of the shape of `b`, meaning it can have as many dimensions as `b` or
 * less, but the sizes of the dimensions must be the same as the first of the
 * shape of `b`.
 * E.g.
 *
 * `findex_set([[0, 1], [2, 3], [4, 5], [6, 7]],
 *             [[4, 5], [6, 7], [8, 9]], [0, 0, 2]) =
 *  [[10, 12], [2, 3], [8, 9], [6, 7]]`
 *
 * `findex_set([[0, 1], [2, 3], [4, 5], [6, 7]],
 *             [[4, 5], [6, 7], [8, 9], [10, 11]],
 *             [[-1, 0], [1, 1], [1, 0], [1, -1]]) =
 *  [[5, 1], [2, 13], [9, 8], [6, 10]]`
 */
FGraphNode *findex_set(FGraphNode *a, FGraphNode *b, FGraphNode *indices);

/**
 * Moves a window view with size `size` along the original Tensor by starting
 * with aligning the first element of the view with the first element of the
 * tensor, copying the elements of the view and moving the window by the step
 * size given for each dimension (the window is first moved in the innermost
 * dimension and after each is iterated moves it in the outer dimensions).
 * Each view becomes a new element in a new outer dimension.
 *
 * `fsliding_window([[0, 1], [2, 3], [4, 5], [6, 7]],
 *                 [3, 2], [1, 1]) =
 * [[[0, 1], [2, 3], [4, 5]], [[2, 3], [4, 5], [6, 7]]]`
 *
 * `fsliding_window([[[0,1,2],[1,2,3],[2,3,4]],
 *                  [[1,2,3],[2,3,4],[3,4,5]],
 *                  [[2,3,4],[3,4,5],[4,5,6]],
 *                  [[3,4,5],[4,5,6],[5,6,7]]],
 *                  [2, 2, 2], [2, 1, 2]) =
 * [[[[0, 1], [1, 2]],
 *   [[1, 2], [2, 3]]],
 *  [[[1, 2], [2, 3]],
 *   [[2, 3], [3, 4]]],
 *  [[[2, 3], [3, 4]],
 *   [[3, 4], [4, 5]]],
 *  [[[3, 4], [4, 5]],
 *   [[4, 5], [5, 6]]]]`
 */
FGraphNode *fsliding_window(FGraphNode *a, const size_t *size,
							const unsigned int *steps);

/**
 * Reprojects the windows (first dimension of `a`) to a common tensor,
 * i.e. if `a = fsliding_window(x, window_size, steps)` `shape` should be the
 * shape of `x`, while steps remain the same and the resulting tensor will have
 * the same shape of `x` with the elements in `a` reprojected to their original
 * position in `x`.
 * Overlapping elements will be summed up. Meaning if e.g. `window_size` and
 *`steps` were the same, the result is exactly `x`.
 * If in a dimension `steps` was larger than `window_size`, the resulting tensor
 * will have `0` elements were the "gaps" between the windows were.
 * If in a dimension `steps` was smaller than `window_size` (the windows were
 * "overlapping") the overlapping elements are summed up in the result.
 *
 * `shape` and `steps` therefore have 1 entry less then `a` has dimensions.
 */
FGraphNode *funslide_window(FGraphNode *a, const size_t *shape,
							const unsigned int *steps);

/**
 * Randomly permutates (=swaps multiple elements with each other without
 * creating, copying or deleting new ones) one axis of the input tensor.
 */
FGraphNode *fpermutate(FGraphNode *a, unsigned int ax);

/**
 * Slides a window along the Tensor and reduces all elements inside that window
 * to their sum (just that one remains in the result tensor), and
 * then slides the window in each dimension by `step_size` forward (like
 * `fsliding_window`). The last dimension is complety pooled, and the result is
 * one dimension smaller then the original tensor
 * - `a` the tensor to pool
 * - `window_size` array with one element less then `a` has dimension, each
 *   describing the window size in that dimension for which for all elements
 *   inside each window the maximum should be taken.
 * - `step_size` array of number of elements the window should be moved after
 *   each pooling for each dimension (one element less then `a` has
 *   dimensions).
 */
FGraphNode *fpooling_sum(FGraphNode *a, const size_t *window_size,
						 const unsigned int *step_size);

/**
 * Slides a window along the Tensor and reduces all elements inside that window
 * to their maximum element (just that one remains in the result tensor), and
 * then slides the window in each dimension by `step_size` forward (like
 * `fsliding_window`). The last dimension is complety pooled, and the result is
 * one dimension smaller then the original tensor
 * - `a` the tensor to pool
 * - `window_size` array with one element less then `a` has dimension, each
 *   describing the window size in that dimension for which for all elements
 *   inside each window the maximum should be taken.
 * - `step_size` array of number of elements the window should be moved after
 *   each pooling for each dimension (one element less then `a` has
 *   dimensions).
 */
FGraphNode *fpooling_max(FGraphNode *a, const size_t *window_size,
						 const unsigned int *step_size);

/**
 * Sets random selected elements in `g` to 0 by the chance `p` (i.e. if `p` is
 * 0.4. each element has a 40% probability of being set to 0).
 */
FGraphNode *fdropout(FGraphNode *g, const double p);
#ifdef __cplusplus
}

// no c++ bindings, but function overloading for c++ header
inline FGraphNode *fconstant(const int value, const size_t *shape,
							 const int dimensions) {
	return fconstant_i(value, shape, dimensions);
}
inline FGraphNode *fconstant(const long value, const size_t *shape,
							 const int dimensions) {
	return fconstant_l(value, shape, dimensions);
}
inline FGraphNode *fconstant(const float value, const size_t *shape,
							 const int dimensions) {
	return fconstant_f(value, shape, dimensions);
}
inline FGraphNode *fconstant(const double value, const size_t *shape,
							 const int dimensions) {
	return fconstant_d(value, shape, dimensions);
}

inline FGraphNode *fadd(FGraphNode *a, FGraphNode *b) { return fadd_g(a, b); }
inline FGraphNode *fadd(FGraphNode *a, const int b) { return fadd_ci(a, b); }
inline FGraphNode *fadd(FGraphNode *a, const long b) { return fadd_cl(a, b); }
inline FGraphNode *fadd(FGraphNode *a, const float b) { return fadd_cf(a, b); }
inline FGraphNode *fadd(FGraphNode *a, const double b) { return fadd_cd(a, b); }

inline FGraphNode *fsub(FGraphNode *a, FGraphNode *b) { return fsub_g(a, b); }
inline FGraphNode *fsub(FGraphNode *a, const int b) { return fsub_ci(a, b); }
inline FGraphNode *fsub(FGraphNode *a, const long b) { return fsub_cl(a, b); }
inline FGraphNode *fsub(FGraphNode *a, const float b) { return fsub_cf(a, b); }
inline FGraphNode *fsub(FGraphNode *a, const double b) { return fsub_cd(a, b); }

inline FGraphNode *fmul(FGraphNode *a, FGraphNode *b) { return fmul_g(a, b); }
inline FGraphNode *fmul(FGraphNode *a, const int b) { return fmul_ci(a, b); }
inline FGraphNode *fmul(FGraphNode *a, const long b) { return fmul_cl(a, b); }
inline FGraphNode *fmul(FGraphNode *a, const float b) { return fmul_cf(a, b); }
inline FGraphNode *fmul(FGraphNode *a, const double b) { return fmul_cd(a, b); }

inline FGraphNode *fdiv(FGraphNode *a, FGraphNode *b) { return fdiv_g(a, b); }
inline FGraphNode *fdiv(FGraphNode *a, const int b) { return fdiv_ci(a, b); }
inline FGraphNode *fdiv(FGraphNode *a, const long b) { return fdiv_cl(a, b); }
inline FGraphNode *fdiv(FGraphNode *a, const float b) { return fdiv_cf(a, b); }
inline FGraphNode *fdiv(FGraphNode *a, const double b) { return fdiv_cd(a, b); }

inline FGraphNode *fdiv(const int a, FGraphNode *b) { return fdiv_ici(a, b); }
inline FGraphNode *fdiv(const long a, FGraphNode *b) { return fdiv_icl(a, b); }
inline FGraphNode *fdiv(const float a, FGraphNode *b) { return fdiv_icf(a, b); }
inline FGraphNode *fdiv(const double a, FGraphNode *b) {
	return fdiv_icd(a, b);
}

inline FGraphNode *fpow(FGraphNode *a, FGraphNode *b) { return fpow_g(a, b); }
inline FGraphNode *fpow(FGraphNode *a, const int b) { return fpow_ci(a, b); }
inline FGraphNode *fpow(FGraphNode *a, const long b) { return fpow_cl(a, b); }
inline FGraphNode *fpow(FGraphNode *a, const float b) { return fpow_cf(a, b); }
inline FGraphNode *fpow(FGraphNode *a, const double b) { return fpow_cd(a, b); }

inline FGraphNode *fmin(FGraphNode *a, FGraphNode *b) { return fmin_g(a, b); }
inline FGraphNode *fmin(FGraphNode *a, const int b) { return fmin_ci(a, b); }
inline FGraphNode *fmin(FGraphNode *a, const long b) { return fmin_cl(a, b); }
inline FGraphNode *fmin(FGraphNode *a, const float b) { return fmin_cf(a, b); }
inline FGraphNode *fmin(FGraphNode *a, const double b) { return fmin_cd(a, b); }

inline FGraphNode *fmax(FGraphNode *a, FGraphNode *b) { return fmax_g(a, b); }
inline FGraphNode *fmax(FGraphNode *a, const int b) { return fmax_ci(a, b); }
inline FGraphNode *fmax(FGraphNode *a, const long b) { return fmax_cl(a, b); }
inline FGraphNode *fmax(FGraphNode *a, const float b) { return fmax_cf(a, b); }
inline FGraphNode *fmax(FGraphNode *a, const double b) { return fmax_cd(a, b); }

inline FGraphNode *fless(FGraphNode *a, FGraphNode *b) { return fless_g(a, b); }
inline FGraphNode *fless(FGraphNode *a, const int b) { return fless_ci(a, b); }
inline FGraphNode *fless(FGraphNode *a, const long b) { return fless_cl(a, b); }
inline FGraphNode *fless(FGraphNode *a, const float b) {
	return fless_cf(a, b);
}
inline FGraphNode *fless(FGraphNode *a, const double b) {
	return fless_cd(a, b);
}

inline FGraphNode *fequal(FGraphNode *a, FGraphNode *b) {
	return fequal_g(a, b);
}
inline FGraphNode *fequal(FGraphNode *a, const int b) {
	return fequal_ci(a, b);
}
inline FGraphNode *fequal(FGraphNode *a, const long b) {
	return fequal_cl(a, b);
}
inline FGraphNode *fequal(FGraphNode *a, const float b) {
	return fequal_cf(a, b);
}
inline FGraphNode *fequal(FGraphNode *a, const double b) {
	return fequal_cd(a, b);
}

inline FGraphNode *fgreater(FGraphNode *a, FGraphNode *b) {
	return fgreater_g(a, b);
}
inline FGraphNode *fgreater(FGraphNode *a, const int b) {
	return fgreater_ci(a, b);
}
inline FGraphNode *fgreater(FGraphNode *a, const long b) {
	return fgreater_cl(a, b);
}
inline FGraphNode *fgreater(FGraphNode *a, const float b) {
	return fgreater_cf(a, b);
}
inline FGraphNode *fgreater(FGraphNode *a, const double b) {
	return fgreater_cd(a, b);
}

inline FGraphNode *fflatten(FGraphNode *a, int dimension) {
	return fflatten_dimension(a, dimension);
}

#include <string>
inline void flogging(FLogType type, std::string msg) {
	flogging(type, msg.c_str());
}
#endif // __cplusplus
#endif // FLINT_H
