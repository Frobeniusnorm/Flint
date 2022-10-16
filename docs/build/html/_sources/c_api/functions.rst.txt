*********
Functions
*********

.. default-domain:: c

Graph and Execution Functions 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
flintInit()
"""""""""""

.. c:function:: void flintInit(int cpu, int gpu)

   :param cpu: 1 if the cpu backend should be initialized, 0 if not
   :param gpu: 1 if the gpu backend should be initialized, 0 if not.

.. c:function:: void flintInit_cpu()
.. c:function:: void flintInit_gpu()

Initializes the cpu and the gpu backend. they are already implicitly called by the execution functions if necessary.
the first method allows disabling of the gpu backend (by passing 0 to its gpu parameter), the cpu backend cannot be
disabled (if a 0 is passed to the cpu backend in the first method, it may still be started by :func:`fExecuteGraph`, if it decides 
that the cpu backend should be chosen). only use those functions if you...

* ...want to explicitly decide where and when the initialization should take place
* ...want to only start one backend

|

flintCleanup()
""""""""""""""

.. c:function:: void flintCleanup()
.. c:function:: void flintCleanup_cpu()
.. c:function:: void flintCleanup_gpu()

Deallocates any resourced allocated by the corresponding backends. 
The first method calls the other two which are only executed if the framework was
initialized, else they do nothing.

|

fSetLoggingLevel()
""""""""""""""""""

.. c:function:: void fSetLoggingLevel(int level)

  :param level: Logging level, between 0 and 4
   
  Sets the logging level of the framework. Adjust this for debugging purposes, or if you release software in which Flint is contained.


  .. seealso::

    :func:`flog`, :enum:`FLogType`   

  Levels:

  * 0: No logging
  * 1: Only :var:`F_ERROR`
  * 2: Logging level 1 + :var:`F_WARNING` (should be used for production)
  * 3: Logging level 2 + :var:`F_INFO` (for developement)
  * 4: Logging level 3 + :var:`F_VERBOSE` (for library developement)
  * 5: Logging level 4 + :var:`F_DEBUG` (when a bug in the library has been found)

flog()
""""""

.. c:function:: void flog(FLogType type, const char* msg)

  Logs a :var:`NULL` terminated string with the given logging level 


fCreateGraph()
"""""""""""""""
.. c:function:: FGraphNode* fCreateGraph(const void* data, const int num_entries, const FType data_type, const size_t* shape, const int dimensions)

  :param data: pointer to the flattened data array that should be loaded into the node 
  :param num_entries: the number of elements (NOT BYTES!) that should be loaded
  :param data_type: the datatype of :var:`data`
  :param shape: an array of size :var:`dimensions`, each entry describing the size of the corresponding dimension. Make sure, :var:`data` 
    is at least as large as the product of all entries in :var:`shape`
  :param dimensions: the number of dimensions

  Creates a Graph with a single store instruction, the data is copied to intern
  memory, so after return of the function, :var:`data` does not have to stay valid. :var:`shape` is copied as well. 

fFreeGraph()
""""""""""""
.. c:function:: void fFreeGraph(FGraphNode* graph)

  :param graph: the graph data that should be released

  Decrements :member:`FGraphNode.reference_counter` of :var:`graph` and deallocates the node and its corresponding data, if the counter becomes 0.
  If the node is deallocated, the same process is repeated with its predecessors.
  So you can safely connect nodes multiple times and have only to free the leaf nodes (i.e. the results), without caring about
  cross-reference, since thouse are handles by the reference counting system.

fCopyGraph()
""""""""""""
.. c:function:: FGraphNode *fCopyGraph(const FGraphNode* graph)

  Copies the graph node, the corresponding operation and additional data and the predecessors (their :member:`FGraphNode.reference_counter` is incremented)

fExecuteGraph()
""""""""""""""""
.. c:function:: FGraphNode* fExecuteGraph(FGraphNode* node)
.. c:function:: FGraphNode* fExecuteGraph_cpu(FGraphNode* node)
.. c:function:: FGraphNode* fExecuteGraph_gpu(FGraphNode* node)

Executes the graph node operations from all yet to be executed predecessors to :var:`node` and returns a node with a :struct:`FResultData` operation
in which the resulting data is stored.
If the graph is executed by the GPU backend, a opencl kernel containing all selected operations is compiled and executed. The kernels are cashed, so it improves the performance
of a program if the same graph-structures are reused (not necessary the same nodes, but the same combination of nodes), since then the backend can 
reuse already compiled kernels. If the CPU backend is chosen, it does not matter, since every operation is executed independently.
The first method selects a backend by basic heuristics of the nodes (like operation types and number of entries), the other two execute the graph with the corresponding backend.
Although the CPU backend uses a thread pool, the method itself is called sequential and waits until the result is computed, so you do not have to synchronize anything.

.. note:: In the future eager execution may be implemented which may improve performance for the cpu backend everytime and of the gpu backend if one uses frequently changing graph structures

|

Operations
^^^^^^^^^^
fadd()
""""""
.. c:function:: FGraphNode *fadd_g(FGraphNode* a, FGraphNode* b)
.. c:function:: FGraphNode *fadd_ci(FGraphNode* a, const int b)
.. c:function:: FGraphNode *fadd_cl(FGraphNode* a, const long b)
.. c:function:: FGraphNode *fadd_cf(FGraphNode* a, const float b)
.. c:function:: FGraphNode *fadd_cd(FGraphNode* a, const double b)

Elementwise addition of a and b :math:`a+b`.

|

fsub()
""""""
.. c:function:: FGraphNode *fsub_g(FGraphNode* a, FGraphNode* b)
.. c:function:: FGraphNode *fsub_ci(FGraphNode* a, const int b)
.. c:function:: FGraphNode *fsub_cl(FGraphNode* a, const long b)
.. c:function:: FGraphNode *fsub_cf(FGraphNode* a, const float b)
.. c:function:: FGraphNode *fsub_cd(FGraphNode* a, const double b)

Elementwise subtraction of a and b: :math:`a-b`.

|

fmul()
""""""
.. c:function:: FGraphNode *fmul_g(FGraphNode* a, FGraphNode* b)
.. c:function:: FGraphNode *fmul_ci(FGraphNode* a, const int b)
.. c:function:: FGraphNode *fmul_cl(FGraphNode* a, const long b)
.. c:function:: FGraphNode *fmul_cf(FGraphNode* a, const float b)
.. c:function:: FGraphNode *fmul_cd(FGraphNode* a, const double b)

Elementwise multiplication of a and b: :math:`a\cdot b`.

|

fdiv()
""""""
.. c:function:: FGraphNode *fdiv_g(FGraphNode* a, FGraphNode* b)
.. c:function:: FGraphNode *fdiv_ci(FGraphNode* a, const int b)
.. c:function:: FGraphNode *fdiv_cl(FGraphNode* a, const long b)
.. c:function:: FGraphNode *fdiv_cf(FGraphNode* a, const float b)
.. c:function:: FGraphNode *fdiv_cd(FGraphNode* a, const double b)

Elementwise division of a and b: :math:`\frac{a}{b}`.

|

fpow()
""""""
.. c:function:: FGraphNode *fpow_g(FGraphNode* a, FGraphNode* b)
.. c:function:: FGraphNode *fpow_ci(FGraphNode* a, const int b)
.. c:function:: FGraphNode *fpow_cl(FGraphNode* a, const long b)
.. c:function:: FGraphNode *fpow_cf(FGraphNode* a, const float b)
.. c:function:: FGraphNode *fpow_cd(FGraphNode* a, const double b)

Takes the elementwise power of a to b: :math:`a^b`.

|

fmatmul()
"""""""""
.. c:function:: FGraphNode *fmatmul(FGraphNode** a, FGraphNode** b)

  Carries out matrix multiplication on the last two dimensions of the tensors. E.g. a matrix multiplication of two tensors with shapes 
  (64, 32, 16) and (16, 24) will yield a tensor with shape (64, 32, 24). Since for one entry of the tensor multiple other previous entries are 
  needed, the operand tensors need to be executed first. Therefor the method will implicitly (or eagerly) execute the two parameter nodes if their
  data is not allready present, the given pointers will be overwritten with the results. 

|

fflatten()
""""""""""
.. c:function:: FGraphNode* fflatten(FGraphNode* a)
.. c:function:: FGraphNode* fflatten_dimension(FGraphNode* a, int dimension)

The first method flattens the complete tensor to a tensor with one dimension, the second method flattens the tensor with :math:`n` dimensions 
along :c:var:`dimension`, resulting in a tensor with :math:`n-1` dimensions. Flattening a dimension will remove it from the shape of the tensor, therefor its 
not possible to flatten the dimension 0.
A Tensor :math:`[[[3, 1, 4], [2, 1, 5]], [[0, 4, 2], [4, 7, 9]]]` flattened along dimension 1 will 
result in :math:`[[3,1,4], [2,1,5], [0,4,2], [4,7,9]]`.

|

fconvert()
""""""""""
.. c:function:: FGraphNode* fconvert(FGraphNode* a, FType newtype)

  Converts the data of :var:`a` to the type given by :var:`newtype`

|

freshape()
""""""""""
.. c:function:: FGraphNode* freshape(FGraphNode* a, size_t* newshape, int dimensions)

  :param a: the operand 
  :param newshape: array of length :var:`dimensions`, representing the new shape for each dimension
  :param dimensions: number of dimensions 

  Reshapes the underlying data of the tensor to the new shape. The product of each dimension of the new shape must be the same as the product 
  of the dimensions of the previous shape (i.e. it must describe the same number of entries of the tensor).
