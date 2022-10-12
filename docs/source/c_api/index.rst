C API
=====

.. default-domain:: c

Usage
-----

Since the framework needs to allocate data (e.g. threadpools or the opencl runtime) which should persist during the runtime there is a initialization
and a cleanup method. The initialization method is implictly called during the first execution of a graph if the framework was not yet initialized.
If you want to specify which backends to use or when Flint should be initialized you may call those methods explicitly.

Regardless of that the core concept of the C API is the :c:struct:`FGraphNode`, which represents the application of a operation on (optional) predecessor nodes (the parameters of the operation).
The initial nodes are always either :c:struct:`FStore` or :c:struct:`FResultData` which stores the corresponding input (or output) data.
The input data is copied and the copied data as well as the result data is managed by Flint and freed when the node is freed.
Do not allocate the nodes by yourself, they are allocated by the framework in the corresponding methods.
All graph nodes have a reference counter to count by how many other nodes they are referenced.
Nevertheless the nodes have to be freed by the function :c:func:`fFreeGraph` which decrements the reference counter by a node 
and frees it if the counter is zero. It recursively repeats this for its predecessors (so if a graph is only referenced by one single result node,
the complete graph is freed with the node), this works only because the graph is azyclic.

Once you have created your input data (:c:func:`fCreateGraph`) and connected it with the desired operations you may pass it to :c:func:`fExecuteGraph`, which returns a 
graph node with :c:struct:`FResultData`.

.. toctree::
  core
