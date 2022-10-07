****************
Structs and Data
****************


.. default-domain:: c

Here you find the relevant data structures used by the C-API:

Graph Structures
^^^^^^^^^^^^^^^^

FGraphNode
""""""""""

.. c:struct:: FGraphNode

  .. c:member:: int num_predecessor

  .. c:member:: FGraphNode** predecessors

  .. c:member:: FOperation* operation 

  .. c:member:: size_t reference_counter

Describes one node in the Graph. Stores the corresponding operation in :member:`FGraphNode.operation`, 
an array of predecessors (the arguments of the operation) in :member:`FGraphNode.predecessors`, 
its size in :member:`FGraphNode.num_predecessor` and the reference counter in `FGraphNode.reference_counter`. 
Do not modify any parameter by yourself, since the framework manages them,
but you can read the data and structure from them. The nodes are allocated by the operation functions, they and their members 
should neither be manually created, edited or freed except by the corresponding flint methods.

FOperation 
""""""""""
.. c:struct:: FOperation 
   
  .. c:member:: int dimensions

  .. c:member:: size_t* shape

  .. c:member:: FOperationType op_type

  .. c:member:: FType data_type

  .. c:member:: void* additional_data

Describes one operation. An operation always has a shape, described by :member:`FOperation.shape` which is an array of size :member:`FOperation.dimensions` 
with each entry denoting the size of the corresponding dimension.
:member:`FOperation.op_type` denotes the type of operation, :member:`FOperation.data_type` the type of the underlying data,
:member:`FOperation.additional_data` is operation specific.

Enums
^^^^^

FOperationType
""""""""""""""
.. c:enum:: FOperationType

Containting each operation type.

FType 
"""""
.. c:enum:: FType

The 4 allowed data types: :var:`INT32` (integer, 32bit), :var:`INT64` (integer, 64bit), :var:`FLOAT32` (floating point, 32bit), 
:var:`FLOAT64` (floating point, 64bit)

Additional Operation Data 
^^^^^^^^^^^^^^^^^^^^^^^^^
Referenced by :member:`additional_data` in :struct:`FOperation`, must be casted, the operation type can be found in :member:`FOperation.op_type`

FResultData
"""""""""""
.. c:struct:: FResultData

  .. c:member:: cl_mem mem_id

  .. c:member:: void* data 

  .. c:member:: size_t num_entries


Stores the resulting data after an execution of :func:`fExecuteGraph`. 
The data can be found in :c:member:`FResultData.data`, the datatype in :c:member:`FOperation.data_type` of the corresponding :struct:`FGraphNode`.
The number of entries (**not** number of bytes) is stored in :member:`FResultData.num_entries`. 
The data may be consistently modified if...

* ...when the data size is changed, num_entries is equivalently updated and c:func:`realloc` is used

* ...the data was not already loaded to the gpu (i.e. the result must be the return value of :func:`fExecuteGraph_cpu`)

FStore
""""""
.. c:struct:: FStore 
  
  .. c:member:: cl_mem mem_id
  
  .. c:member:: void* data

  .. c:member:: size_t num_entries

Result of an call to :func:`fCreateGraph`, see :struct:`FResultData`.
Data of this Operation may always be changed, since the framework assumes this.

.. warning::

  This will change in the future, there will be a "dirty bit" in this struct and a function to update its data

FConst
""""""
.. c:struct:: FConst 
  
  .. c:member:: void* value 

Stores a single constant. The underlying data may also be changed between executions.

FSlice
""""""
.. c:struct:: FSlice 

  .. c:member:: size_t* start

  .. c:member:: size_t* end

  .. c:member:: size_t* step

Represents one slice operation. Each of the members has one entry for each dimension. 

.. seealso::

  :func:`fslice` and :func:`fslice_step` 
