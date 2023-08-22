#include "flint.h"
#include "flint_helper.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <sys/types.h>
#include <tuple>
#include <vector>
/**
 * The multi dimensional implementation of `Tensor`.
 */
template <typename T, unsigned int n> struct Tensor {
  template <typename K, unsigned int k> friend struct Tensor;
  friend struct Flint;
  // storage type is the vector of the recursive type
  typedef std::vector<typename Tensor<T, n - 1>::storage_type> storage_type;
  typedef std::initializer_list<typename Tensor<T, n - 1>::init_type> init_type;
  /**
   * Uninitialized Tensor
   */
  Tensor() : node(nullptr) {}
  /**
   * Creates a Tensor from a `n`-times nested `std::initializer_list`
   * (`init_type` is a recursive defined type definition).
   * E.g.
   *
   * @code{
   * Tensor<float, 2> t1{{-1., 0.}, {1., 2.}};
   * Tensor<float, 3> t2 = {{{0, 1}, {1, 2}}, {{3, 4}, {5, 6}}};
   * }
   */
  Tensor(init_type data) {
    isTensorType<T>();
    static_assert(n > 1, "Dimension must be at least 1");
    initShape(data, 0);
    std::vector<T> flat = FLINT_HPP_HELPER::flattened<T>(data);
    node = fCreateGraph(flat.data(), flat.size(), toFlintType<T>(),
                        shape.data(), shape.size());
    // the node which is currently hold is always referenced
    node->reference_counter = 1;
  }
  /**
   * Creates a Tensor from a `n`-times nested `std::vector`
   * (`storage_type` is a recursive defined type definition).
   * E.g.
   *
   * @code{
   * std::vector<std::vector<float>> s1 = {{-1., 0.}, {1., 2.}};
   * Tensor<float, 2> t1(s1);
   * }
   */
  Tensor(storage_type data) {
    isTensorType<T>();
    static_assert(n > 1, "Dimension must be at least 1");
    initShape(data, 0);
    std::vector<T> flat = FLINT_HPP_HELPER::flattened<T>(data);
    node = fCreateGraph(flat.data(), flat.size(), toFlintType<T>(),
                        shape.data(), shape.size());
    // the node which is currently hold is always referenced
    node->reference_counter = 1;
  }
  /**
   * Constructs a Tensor directly from a `FGraphNode` and a shape
   */
  Tensor(FGraphNode *node, std::array<size_t, n> shape)
      : node(node), shape(shape) {
    // here data is still fine
    total_size = 1;
    for (int ds : shape)
      total_size *= ds;
    node->reference_counter++;
    fOptimizeMemory(node); // should be legal here, since C++ header adjust
                           // reference_counter
  }
  /**
   * Copy constructor. Copies the underlying Graph structure by creating a new
   * node with the same operation, shape and data types. The new predecessor
   * array points to the same predecessors (memory safety is ensured with
   * reference counting).
   *
   * If `other` has result data or if it is a storage node, the complete CPU
   * data is directly copied. Since this operation is expensive it is advised to
   * only use it if it is completly necessary.
   */
  Tensor(const Tensor &other) {
    shape = other.shape;
    total_size = other.total_size;
    node = fCopyGraph(other.node);
    node->reference_counter++;
  }
  /**
   * Copy operator. Copies the underlying Graph structure by creating a new
   * node with the same operation, shape and data types. If there was any
   * previous allocated operation node allocated by this Tensor it is cleaned
   * up. The new predecessor array points to the same predecessors (memory
   * safety is ensured with reference counting).
   *
   * If `other` has result data or if it is a storage node, the complete CPU
   * data is directly copied. Since this operation is expensive it is advised to
   * only use it if it is completly necessary.
   */
  Tensor<T, n> &operator=(const Tensor &other) {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
    shape = other.shape;
    total_size = other.total_size;
    node = fCopyGraph(other.node);
    node->reference_counter++;
    return *this;
  }
  /**
   * Move constructor. Moves every important field from `other` to this Tensor.
   * `other` is invalidated after this operation.
   */
  Tensor(Tensor &&other) {
    shape = other.shape;
    total_size = other.total_size;
    node = other.node; // was held by previous tensor -> no increment necessary
    other.node = nullptr;
  }
  /**
   * Returns the shape of this Tensor as a array with `n` entries.
   * Each entry describes the size of the corresponding dimension.
   * E.g.
   *
   * @code{
   * Tensor<float, 2> t1{{-1., 0.}, {1., 2.}};
   * std::array<size_t, 2> shape1 = t1.get_shape();
   * // shape1 = {2, 2}
   * }
   */
  std::array<size_t, n> get_shape() const { return shape; }
  /**
   * Move operator. Moves every important field from `other` to this Tensor.
   * `other` is invalidated after this operation. If there was any previous
   * allocated operation node allocated by this Tensor it is cleaned up.
   */
  Tensor<T, n> &operator=(Tensor &&other) {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
    shape = other.shape;
    total_size = other.total_size;
    node = other.node;
    other.node = nullptr;
    return *this;
  }
  /**
   * Cleans up this tensor and frees all underlying data by reference counting.
   */
  ~Tensor() {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
  }
  /**
   * Retrieves the data of the current node and converts it into a
   * multidimensional vector. Executes the node if necessary (if it was not
   * executed prior). This operation has to duplicate the complete data.
   * Since that is a memory heavy and slow operation, it is recommended to use
   * the index operator `operator[]` whenever possible instead.
   * E.g.
   *
   * @code{
   * Tensor<int, 3> foo = Tensor<int, 3>::constant(42, 2, 2, 1);
   * std::vector<std::vector<std::vector<int>>> foo_res = *foo;
   * // foo_res = {{{42}, {42}}, {{42}, {42}}}
   * }
   */
  storage_type operator*() {
    if (!node->result_data)
      execute();
    if (!node->result_data->data)
      fSyncMemory(node);
    FResultData *store = node->result_data;
    storage_type result(shape[0]);
    const std::vector<T> src =
        std::vector<T>((T *)store->data, (T *)store->data + store->num_entries);
    bringIntoShape(result, src, 0, 0, total_size);
    return result;
  }
  /**
   * Serializes the underlying data of the Tensor to a binary vector.
   * If the Tensor has no Result Data it is executed.
   */
  std::vector<char> serialize() {
    size_t no_bytes;
    char *data = fserialize(node, &no_bytes);
    const std::vector<char> foo(data, data + no_bytes);
    free(data);
    return foo;
  }
  /**
   * Deserializes the binary representation of Tensor data back to a Tensor
   * object.
   */
  static Tensor<T, n> deserialize(char *data) {
    FGraphNode *node = fdeserialize(data);
    if (n != node->operation.dimensions)
      flogging(F_ERROR, "Deserializing data of a " +
                            std::to_string(node->operation.dimensions) +
                            " dimensional Tensor into a " + std::to_string(n) +
                            " dimensional Tensor is not possible!");
    if (toFlintType<T>() != node->operation.data_type)
      flogging(F_ERROR,
               "Deserializing data of a " +
                   FLINT_HPP_HELPER::typeString(node->operation.data_type) +
                   " Tensor into a " +
                   FLINT_HPP_HELPER::typeString(toFlintType<T>()) +
                   " Tensor is not possible!");
    std::array<size_t, n> shape;
    for (int i = 0; i < node->operation.dimensions; i++)
      shape[i] = node->operation.shape[i];
    return Tensor<T, n>(node, shape);
  }
  /**
   * Deserializes the binary representation of Tensor data back to a Tensor
   * object.
   */
  static Tensor<T, n> deserialize(std::vector<char> data) {
    return deserialize(data.data());
  }
  /**
   * Executes the underlying operation (and lazily the operations of the parents
   * if needed) if it was not already executed prior (in that case the operation
   * does nothing).
   * If Flint was initiallized implicitly (without ever calling `flintInit`) or
   * with `FLINT_BACKEND_BOTH` the backend is chosen automatically by heuristics
   * and initialized if it was not prior.
   */
  void execute() {
    if (!node->result_data || !node->result_data->data) {
      node = fOptimizeMemory(fExecuteGraph(node));
    }
  }
  /**
   * Executes the underlying operation (and lazily the operations of the parents
   * if needed) if it was not already executed prior (in that case the operation
   * does nothing).
   * Uses the CPU backend and initializes it if it was not initialized.
   */
  void execute_cpu() {
    if (!node->result_data || !node->result_data->data) {
      node = fOptimizeMemory(fExecuteGraph_cpu(node));
    }
  }
  /**
   * Executes the underlying operation (and lazily the operations of the parents
   * if needed) if it was not already executed prior (in that case the operation
   * does nothing).
   * Uses the CPU backend and initializes it if it was not initialized.
   */
  void execute_gpu() {
    if (!node->result_data || !node->result_data->data) {
      node = fOptimizeMemory(fExecuteGraph_gpu(node));
    }
  }
  /**
   * Convenience Method that calls `execute` and returns the Tensor object
   * (the same, no new node is created!).
   */
  Tensor<T, n> &operator()() {
    execute();
    return *this;
  }
  /**
   * Negates the elements of this Tensor.
   * E.g.
   *
   * @code{
   * Tensor<float, 2> foo = {{-3, 3.141592}, {42.0798, -4.3}};
   * std::cout << (-foo)() << std::endl;
   * // Tensor<FLOAT32, shape: [2, 2]>(
   * //  [[3.000000, -3.141592],
   * //   [-42.079800, 4.300000]])
   * }
   */
  Tensor<T, n> operator-() const { return Tensor<T, n>(fneg(node), shape); }
  /**
   * Returns a tensor `x` with the shape of a with `x[i] = 1` if `a[i] >= 0`
   * else `x[i] = -1`. If you need to distinguish additionally for 0 values,
   * take a look at `equal`. E.g.
   *
   * @code{
   * Tensor<float, 2> foo = {{-3, 3.141592}, {42.0798, -4.3}};
   * std::cout << (foo.sign())() << std::endl;
   * // Tensor<INT32, shape: [2, 2]>(
   * //  [[-1, 1],
   * //   [1, -1]])
   * }
   */
  Tensor<int, n> sign() const { return Tensor<int, n>(fsign(node), shape); }
  /**
   * Returns a int tensor `x` with the shape of `this` with `x[i] = 1` if
   * `this[i] % 2 = 0` else `x[i] = 0`. This Tensor needs to have a integer
   * type. E.g.
   *
   * @code{
   * Tensor<int, 2> foo = {{2, 3}, {42, 7}};
   * std::cout << (foo.even())() << std::endl;
   * // Tensor<INT32, shape: [2, 2]>(
   * //  [[1, 0],
   * //   [1, 0]])
   * }
   */
  Tensor<int, n> even() const {
    static_assert(std::is_same<T, int>() || std::is_same<T, long>());
    return Tensor<int, n>(feven(node), shape);
  }
  /**
   * Indexes the Tensor in its last dimension by `index`.
   * The returned type `TensorView` stores the underlying data of this Tensor
   * and the given index. It is only valid and functional as long as the
   * underlying data of this Tensor is not destructed (i.e. as long as this
   * object is alive or as it is attached as the parent of another Tensor).
   * If the underlying data is not yet computed, executes this Tensor.
   * E.g.
   *
   * @code{
   * Tensor<int, 3> foo{{{0,1}, {2,3}}, {{4,5}, {6,7}}};
   * TensorView<int, 2> bar = foo[1];
   * TensorView<int, 1> baz = bar[1];
   * std::cout << baz[0] << std::endl; // 6
   * std::cout << foo[0][1][1] << std::endl; // 3
   * }
   */
  TensorView<T, n - 1> operator[](const size_t index) {
    if (!node->result_data)
      execute();
    if (!node->result_data->data) {
      fSyncMemory(node);
    }
    FResultData *store = node->result_data;
    size_t alrdindx = index;
    std::vector<size_t> ns(shape.size() - 1);
    for (size_t i = 1; i < shape.size(); i++) {
      ns[i - 1] = shape[i];
      alrdindx *= shape[i];
    }
    return TensorView<T, n - 1>((T *)store->data, ns, alrdindx);
  }
  /**
   * Converts this Tensor to a string representation.
   * If the Tensor was not yet executed, it won't be, instead of the data it
   * will say "<not yet executed>".
   */
  operator std::string() {
    const FOperation op = node->operation;
    std::string foo = "Tensor<" +
                      (op.data_type == F_INT32     ? std::string("INT32")
                       : op.data_type == F_INT64   ? std::string("INT64")
                       : op.data_type == F_FLOAT32 ? std::string("FLOAT32")
                                                   : std::string("FLOAT64")) +
                      ", shape: " + FLINT_HPP_HELPER::arrayString(shape) + ">(";
    if (op.op_type != FSTORE && !node->result_data)
      foo += "<not yet executed>";
    else {
      foo += "\n" + FLINT_HPP_HELPER::vectorString(this->operator*(), " ");
    }
    foo += ")";
    return foo;
  }
  /**
   * Calls `serialize` on this Tensor and pipes the returned data to the
   * stream.
   */
  friend std::ofstream &operator<<(std::ofstream &os, Tensor<T, n> &t) {
    for (char c : t.serialize()) {
      os.put(c);
    }
    return os;
  }
  static Tensor<T, n> read_from(std::ifstream &is) {
    // read to shape
    std::vector<char> data(4 + sizeof(FType) + sizeof(int));
    is.read(data.data(), data.size());
    size_t index = 4 + sizeof(FType);
    int dimensions = 0;
    for (int i = sizeof(int) - 1; i >= 0; i--)
      dimensions |= data[index++] << (i * 8);
    // get shape data
    size_t prev_size = data.size();
    data.resize(data.size() + dimensions * sizeof(size_t));
    is.read(data.data() + prev_size, dimensions * sizeof(size_t));
    std::array<size_t, n> nshape;
    size_t total_size = 1;
    for (int i = 0; i < dimensions; i++) {
      size_t shape = 0;
      for (int j = sizeof(size_t) - 1; j >= 0; j--) {
        shape |= data[index++] << (j * 8);
      }
      nshape[i] = shape;
      total_size *= shape;
    }
    // get actual data
    prev_size = data.size();
    data.resize(prev_size + total_size * sizeof(T));
    is.read(data.data() + prev_size, total_size * sizeof(T));
    return deserialize(data.data());
  }
  /**
   * Calls `std::string()` on this Tensor and pipes the returned string to the
   * pipe.
   */
  friend std::ostream &operator<<(std::ostream &os, Tensor<T, n> &t) {
    os << (std::string)t;
    return os;
  }
  // to calculate the return type of two tensors at compile time
  template <typename K>
  using stronger_return =
      typename std::conditional<isStronger<K, T>(), K, T>::type;
  /**
   * Elementwise addition of this Tensor and `other`. If the dimensions differ
   * the smaller Tensor is broadcasted along the first dimensions which are not
   * shared of the larger one. The datatype of the result is the datatype with
   * higher precedence. E.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0,1}, {2,3}}, {{4,5}, {6,7}}};
   * Tensor<float, 2> b{{4,2},{0.5f,1}};
   * std::cout << (a + b)() << std::endl;
   * // Tensor<FLOAT32, shape: [2, 2, 2]>(
   * // [[[4.000000, 3.000000],
   * //   [2.500000, 4.000000]],
   * //  [[8.000000, 7.000000],
   * //   [6.500000, 8.000000]]])
   * }
   * */
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  operator+(const Tensor<K, k> &other) const {
    if constexpr (k >= n)
      return Tensor<stronger_return<K>, k>(fadd(node, other.node), other.shape);
    else
      return Tensor<stronger_return<K>, n>(fadd(node, other.node), shape);
  }
  /**
   * Elementwise addition of the constant `other` to this Tensor.
   * If the datatype of `K` is stronger (stronger precedence) than the datatype
   * of this Tensor `T`, `K` will be the result type, else `T`.
   */
  template <typename K>
  Tensor<stronger_return<K>, n> operator+(const K other) const {
    return Tensor<stronger_return<K>, n>(fadd(node, other), shape);
  }
  /**
   * Elementwise subtraction of this Tensor and `other`. If the dimensions
   * differ the smaller Tensor is broadcasted along the first dimensions which
   * are not shared of the larger one. The datatype of the result is the
   * datatype with higher precedence. E.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0,1}, {2,3}}, {{4,5}, {6,7}}};
   * Tensor<float, 2> b{{4,2},{0.5f,1}};
   * std::cout << (a - b)() << std::endl;
   * // Tensor<FLOAT32, shape: [2, 2, 2]>(
   * // [[[-4.000000, -1.000000],
   * //   [1.500000, 2.000000]],
   * //  [[0.000000, 3.000000],
   * //   [5.500000, 6.000000]]])
   * }
   * */
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  operator-(const Tensor<K, k> &other) const {
    if constexpr (k >= n)
      return Tensor<stronger_return<K>, k>(fsub(node, other.node), other.shape);
    else
      return Tensor<stronger_return<K>, n>(fsub(node, other.node), shape);
  }
  /**
   * Elementwise substraction of the constant `other` from this Tensor.
   * If the datatype of `K` is stronger (stronger precedence) than the datatype
   * of this Tensor `T`, `K` will be the result type, else `T`.
   */
  template <typename K>
  Tensor<stronger_return<K>, n> operator-(const K other) const {
    return Tensor<stronger_return<K>, n>(fsub(node, other), shape);
  }
  /**
   * Elementwise multiplication of this Tensor and `other`. If the dimensions
   * differ the smaller Tensor is broadcasted along the first dimensions which
   * are not shared of the larger one. The datatype of the result is the
   * datatype with higher precedence. E.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0,1}, {2,3}}, {{4,5}, {6,7}}};
   * Tensor<float, 2> b{{4,2},{0.5f,1}};
   * std::cout << (a * b)() << std::endl;
   * // Tensor<FLOAT32, shape: [2, 2, 2]>(
   * // [[[0.000000, 2.000000],
   * //   [1.000000, 3.000000]],
   * //  [[16.000000, 10.000000],
   * //   [3.000000, 7.000000]]])
   * }
   * */
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  operator*(const Tensor<K, k> &other) const {
    if constexpr (k >= n)
      return Tensor<stronger_return<K>, k>(fmul(node, other.node), other.shape);
    else
      return Tensor<stronger_return<K>, n>(fmul(node, other.node), shape);
  }
  /**
   * Elementwise multiplication of the constant `other` from this Tensor.
   * If the datatype of `K` is stronger (stronger precedence) than the datatype
   * of this Tensor `T`, `K` will be the result type, else `T`.
   */
  template <typename K>
  Tensor<stronger_return<K>, n> operator*(const K other) const {
    return Tensor<stronger_return<K>, n>(fmul(node, other), shape);
  }
  /**
   * Elementwise division of this Tensor and `other`. If the dimensions
   * differ the smaller Tensor is broadcasted along the first dimensions which
   * are not shared of the larger one. The datatype of the result is the
   * datatype with higher precedence. E.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0,1}, {2,3}}, {{4,5}, {6,7}}};
   * Tensor<float, 2> b{{4,2},{0.5f,1}};
   * std::cout << (a / b)() << std::endl;
   * // Tensor<FLOAT32, shape: [2, 2, 2]>(
   * // [[[0.000000, 0.500000],
   * //   [4.000000, 3.000000]],
   * //  [[1.000000, 2.500000],
   * //   [12.000000, 7.000000]]])
   * }
   * */
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  operator/(const Tensor<K, k> &other) const {
    if constexpr (k >= n)
      return Tensor<stronger_return<K>, k>(fdiv(node, other.node), other.shape);
    else
      return Tensor<stronger_return<K>, n>(fdiv(node, other.node), shape);
  }
  /**
   * Elementwise division of the constant `other` from this Tensor.
   * If the datatype of `K` is stronger (stronger precedence) than the datatype
   * of this Tensor `T`, `K` will be the result type, else `T`.
   */
  template <typename K>
  Tensor<stronger_return<K>, n> operator/(const K other) const {
    return Tensor<stronger_return<K>, n>(fdiv(node, other), shape);
  }
  /**
   * Flattens the complete tensor to a tensor with one dimension.
   * E.g.
   *
   * @code{
   * Tensor<long, 3> a = {{{3, 1, 4}, {2, 1, 5}}, {{0, 4, 2}, {4, 7, 9}}};
   * std::cout << (a.flattened())() << std::endl;
   * // Tensor<INT64, shape: 12>([3, 1, 4, 2, 1, 5, 0, 4, 2, 4, 7, 9])
   * }
   */
  Tensor<T, 1> flattened() const {
    FGraphNode *foo = fflatten(node);
    return Tensor<T, 1>(foo, total_size);
  }
  /**
   * Flattens this tensor with `n` dimensions along
   * `dimension`, resulting in a tensor with `n-1` dimensions.
   * Flattening a dimension will remove it from the shape of the tensor.
   * The data stays the same, you can imagine the elements along the flattened
   * dimension to be appended to each other. E.g.
   *
   * @code{
   * Tensor<long, 3> a = {{{3, 1, 4}, {2, 1, 5}}, {{0, 4, 2}, {4, 7, 9}}};
   * std::cout << (a.flattened(1))() << std::endl;
   * // Tensor<INT64, shape: [4, 3]>(
   * // [[3, 1, 4],
   * //  [2, 1, 5],
   * //  [0, 4, 2],
   * //  [4, 7, 9]])
   * }
   */
  Tensor<T, n - 1> flattened(const int dimension) const {
    FGraphNode *foo = fflatten_dimension(node, dimension);
    std::array<size_t, n - 1> ns;
    std::copy_n(foo->operation.shape, (size_t)foo->operation.dimensions,
                ns.begin());
    return Tensor<T, n - 1>(foo, ns);
  }
  /**
   * Elementwise power of this Tensor to `other`. If the dimensions
   * differ the smaller Tensor is broadcasted along the first dimensions which
   * are not shared of the larger one. The datatype of the result is the
   * datatype with higher precedence. E.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
   * Tensor<double, 2> b{{4, 2}, {0.5f, 1}};
   * std::cout << (a.pow(b))() << std::endl;
   * // Tensor<FLOAT64, shape: [2, 2, 2]>(
   * // [[[0.000000, 1.000000],
   * //   [1.414214, 3.000000]],
   * //  [[256.000000, 25.000000],
   * //   [2.449490, 7.000000]]])
   * }
   * */
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k >= n ? k : n> pow(const Tensor<K, k> &other) {
    if constexpr (k >= n)
      return Tensor<stronger_return<K>, k>(fpow(node, other.node), other.shape);
    else
      return Tensor<stronger_return<K>, n>(fpow(node, other.node), shape);
  }
  /**
   * Elementwise power of this tensor to the constant `other`.
   * If the datatype of `K` is stronger (stronger precedence) than the datatype
   * of this Tensor `T`, `K` will be the result type, else `T`.
   */
  template <typename K> Tensor<stronger_return<K>, n> pow(const K other) {
    return Tensor<stronger_return<K>, n>(fpow(node, other), shape);
  }
  /**
   * Takes the elementwise natural logarithm of this Tensor.
   */
  Tensor<to_float<T>, n> log() {
    return Tensor<to_float<T>, n>(flog(node), shape);
  }
  /**
   * Takes the elementwise logarithm dualis of this Tensor.
   */
  Tensor<to_float<T>, n> log2() {
    return Tensor<to_float<T>, n>(flog2(node), shape);
  }
  /**
   * Takes the elementwise logarithm to basis 10 of this Tensor.
   */
  Tensor<to_float<T>, n> log10() {
    return Tensor<to_float<T>, n>(flog10(node), shape);
  }
  /**
   * Takes the elementwise square root of this Tensor.
   */
  Tensor<to_float<T>, n> sqrt() {
    return Tensor<to_float<T>, n>(fsqrt_g(node), shape);
  }
  /**
   * Takes the elementwise exponent of this Tensor (power of the constant `e` to
   * this Tensor).
   */
  Tensor<to_float<T>, n> exp() {
    return Tensor<to_float<T>, n>(fexp(node), shape);
  }
  /**
   * Takes the elementwise sinus of this Tensor.
   */
  Tensor<to_float<T>, n> sin() {
    return Tensor<to_float<T>, n>(fsin(node), shape);
  }
  /**
   * Takes the elementwise cosinus of this Tensor.
   */
  Tensor<to_float<T>, n> cos() {
    return Tensor<to_float<T>, n>(fcos(node), shape);
  }
  /**
   * Takes the elementwise tangents of this Tensor.
   */
  Tensor<to_float<T>, n> tan() {
    return Tensor<to_float<T>, n>(ftan(node), shape);
  }
  /**
   * Takes the elementwise arcsinus of this Tensor (`sin^(-1)`).
   */
  Tensor<to_float<T>, n> asin() {
    return Tensor<to_float<T>, n>(fasin(node), shape);
  }
  /**
   * Takes the elementwise arccosinus of this Tensor (`cos^(-1)`).
   */
  Tensor<to_float<T>, n> acos() {
    return Tensor<to_float<T>, n>(facos(node), shape);
  }
  /**
   * Takes the elementwise arctangents of this Tensor (`tan^(-1)`).
   */
  Tensor<to_float<T>, n> atan() {
    return Tensor<to_float<T>, n>(fatan(node), shape);
  }
  /**
   * Carries out matrix multiplication on the last two dimensions of the
   * tensors (broadcasts all others). E.g. a matrix multiplication of two
   * tensors with shapes `(64, 32, 16)` and `(16, 24)` will yield a tensor with
   * shape `(64, 32, 24)`.
   *
   * Since for one entry of the tensor multiple other
   * previous entries are needed, the operand tensors need to be executed first.
   * Therefor the method will implicitly (or eagerly) execute this Tensor and
   * `other` if their data is not allready present. E.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0, 1},
   *                   {2, 3}},
   *                  {{4, 5},
   *                   {6, 7}}};
   * Tensor<double, 2> b{{4,    2, 3.5f},
   *                     {0.5f, 1, 0}};
   * std::cout << (a.matmul(b))() << std::endl;
   * // Tensor<FLOAT64, shape: [2, 2, 3]>(
   * // [[[0.500000, 1.000000, 0.000000],
   * //   [9.500000, 7.000000, 7.000000]],
   * //  [[18.500000, 13.000000, 14.000000],
   * //   [27.500000, 19.000000, 21.000000]]])
   * }*/
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k >= n ? k : n> matmul(Tensor<K, k> &other) {
    int x = shape[shape.size() - 2];
    int z = other.shape[other.shape.size() - 1];
    std::array<size_t, k >= n ? k : n> ns;
    for (size_t i = 0; i < ns.size() - 2; i++) {
      ns[i] = k >= n ? other.shape[i] : shape[i];
    }
    ns[ns.size() - 2] = x;
    ns[ns.size() - 1] = z;
    return Tensor < stronger_return<K>,
           k >= n ? k : n > (fmatmul(node, other.node), ns);
  }
  /**
   * Converts this Tensor (and the underlying data) to type `K` given in the
   * template. `K` must be one of `int`, `long`, `float`, `double`. The data is
   * converted, not reinterpreted.
   */
  template <typename K> Tensor<K, n> convert() const {
    return Tensor<K, n>(fconvert(node, toFlintType<K>()), shape);
  }
  /**
   * Reshapes this Tensor to a new shape with arbitrary dimensions.
   * It can have less dimensions, more dimensions and a completly different
   * shape, the only assumption that has to hold is that the product of the new
   * shape is the same as the product of the old shape (the new shape represents
   * as many elements as the old).
   */
  template <typename... args>
  Tensor<T, sizeof...(args)> reshape(args... shape) {
    constexpr size_t newdim = sizeof...(args);
    std::array<size_t, newdim> new_shape{static_cast<size_t>(shape)...};
    return Tensor<T, newdim>(freshape(node, new_shape.data(), newdim),
                             new_shape);
  }
  /**
   * Reshapes this Tensor to a new shape with arbitrary dimensions.
   * It can have less dimensions, more dimensions and a completly different
   * shape, the only assumption that has to hold is that the product of the new
   * shape is the same as the product of the old shape (the new shape represents
   * as many elements as the old).
   */
  template <int k> Tensor<T, k> reshape_array(std::array<size_t, k> new_shape) {
    return Tensor<T, k>(freshape(node, new_shape.data(), k), new_shape);
  }
  /**
   * Adds a new dimension at an arbitrary position to the tensor and repeats the
   * following dimensions to match a given shape.
   *
   * - `ax` the dimension prior to which the new dimension will be inserted (`0`
   *    means a new dimension in the front, `n + 1` means as a new last
   *    dimension).
   * - `ax_size` the new size of that dimension (repeats the following
   *    dimensions `ax_size - 1` times).
   *
   * @code{
   * Tensor<double, 2> a = {{0, 1}, {2, 3}};
   * std::cout << a.expand(0, 3)() << std::endl;
   * // Tensor<FLOAT64, shape: [3, 2, 2]>(
   * // [[[0.000000, 1.000000],
   * //   [2.000000, 3.000000]],
   * //  [[0.000000, 1.000000],
   * //   [2.000000, 3.000000]],
   * //  [[0.000000, 1.000000],
   * //   [2.000000, 3.000000]]])
   * std::cout << a.expand(1, 3)() << std::endl;
   * // Tensor<FLOAT64, shape: [2, 3, 2]>(
   * // [[[0.000000, 1.000000],
   * //   [0.000000, 1.000000],
   * //   [0.000000, 1.000000]],
   * //  [[2.000000, 3.000000],
   * //   [2.000000, 3.000000],
   * //   [2.000000, 3.000000]]])
   * std::cout << a.expand(2, 3)() << std::endl;
   * // Tensor<FLOAT64, shape: [2, 2, 3]>(
   * // [[[0.000000, 0.000000, 0.000000],
   * //   [1.000000, 1.000000, 1.000000]],
   * //  [[2.000000, 2.000000, 2.000000],
   * //   [3.000000, 3.000000, 3.000000]]])
   * }
   */
  Tensor<T, n + 1> expand(int ax = n, int ax_size = 0) {
    FGraphNode *nn = fexpand(node, ax, ax_size);
    std::array<size_t, n + 1> new_shape;
    std::memcpy(new_shape.data(), nn->operation.shape,
                sizeof(size_t) * (n + 1));
    return Tensor<T, n + 1>(nn, new_shape);
  }
  /** Takes the minimum of this tensor and `other` element wise (the lower value
   * is the result, if one tensor is smaller it will be broadcasted).*/
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  min(const Tensor<K, k> &other) const {
    if constexpr (k >= n)
      return Tensor<stronger_return<K>, k>(fmin(node, other.node), other.shape);
    else
      return Tensor<stronger_return<K>, n>(fmin(node, other.node), shape);
  }
  /** Takes the minimum of this Tensor and the constant value `other` for each
   * element.*/
  template <typename K> Tensor<stronger_return<K>, n> min(const K other) const {
    return Tensor<stronger_return<K>, n>(fmin(node, other), shape);
  }
  /** Takes the maximum of this tensor and `other` element wise (the higher
   * value is the result, if one tensor is smaller it will be broadcasted).*/
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k >= n ? k : n>
  max(const Tensor<K, k> &other) const {
    if constexpr (k >= n)
      return Tensor<stronger_return<K>, k>(fmax(node, other.node), other.shape);
    else
      return Tensor<stronger_return<K>, n>(fmax(node, other.node), shape);
  }
  /** Takes the maximum of this Tensor and the constant value `other` for each
   * element.*/
  template <typename K> Tensor<stronger_return<K>, n> max(const K other) const {
    return Tensor<stronger_return<K>, n>(fmax(node, other), shape);
  }
  /**
   * Compares this tensor and `other` elementwise and returns a 0,1 integer
   * Tensor. `0` denotes that `this >= other`, `1` that `this < other`.
   */
  template <typename K, unsigned int k>
  Tensor<int, k >= n ? k : n> operator<(const Tensor<K, k> &other) const {
    if constexpr (k >= n)
      return Tensor<int, k>(fless(node, other.node), other.shape);
    else
      return Tensor<int, n>(fless(node, other.node), shape);
  }
  /**
   * Compares this tensor and the constant `other` elementwise and returns a 0,1
   * integer Tensor. `0` denotes that `this >= other`, `1` that `this < other`.
   */
  template <typename K> Tensor<int, n> operator<(const K other) const {
    return Tensor<int, n>(fless(node, other), shape);
  }
  /**
   * Compares this tensor and `other` elementwise and returns a 0,1 integer
   * Tensor. `0` denotes that `this <= other`, `1` that `this > other`.
   */
  template <typename K, unsigned int k>
  Tensor<int, k >= n ? k : n> operator>(const Tensor<K, k> &other) const {
    if constexpr (k >= n)
      return Tensor<int, k>(fgreater(node, other.node), other.shape);
    else
      return Tensor<int, n>(fgreater(node, other.node), shape);
  }
  /**
   * Compares this tensor and the constant `other` elementwise and returns a 0,1
   * integer Tensor. `0` denotes that `this <= other`, `1` that `this > other`.
   */
  template <typename K> Tensor<int, n> operator>(const K other) const {
    return Tensor<int, n>(fgreater(node, other), shape);
  }
  /**
   * Compares this tensor and `other` elementwise and returns a 0,1 integer
   * Tensor. `0` denotes that `this != other`, `1` that `this == other`.
   */
  template <typename K, unsigned int k>
  Tensor<int, k >= n ? k : n> equal(const Tensor<K, k> &other) const {
    if constexpr (k >= n)
      return Tensor<int, k>(fequal(node, other.node), other.shape);
    else
      return Tensor<int, n>(fequal(node, other.node), shape);
  }
  /**
   * Compares this tensor and the constant `other` elementwise and returns a 0,1
   * integer Tensor. `0` denotes that `this != other`, `1` that `this == other`.
   */
  template <typename K> Tensor<int, n> equal(const K other) const {
    return Tensor<int, n>(fequal(node, other), shape);
  }
  /** Reduces one dimension of the tensor by additive folding e.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0, 1, 32}, {2, 3, 4}},
   *                  {{4, 5, -6}, {6, 7, -1}}};
   * std::cout << (a.reduce_sum(0))() << std::endl;
   * // Tensor<INT32, shape: [2, 3]>(
   * // [[4, 6, 26],
   * //  [8, 10, 3]])
   * std::cout << (a.reduce_sum(1))() << std::endl;
   * // Tensor<INT32, shape: [2, 3]>(
   * // [[2, 4, 36],
   * //  [10, 12, -7]])
   * std::cout << (a.reduce_sum(2))() << std::endl;
   * // Tensor<INT32, shape: [2, 2]>(
   * // [[33, 9],
   * //  [3, 12]])
   * }
   *
   * The results of this Tensor must be available, to
   * ensure that the method may execute the Tensor. */
  Tensor<T, n - 1> reduce_sum(int dimension) {
    std::array<size_t, n - 1> ns;
    if (dimension < 0)
      dimension = shape.size() + dimension;
    for (int i = 0; i < dimension; i++)
      ns[i] = shape[i];
    for (size_t i = dimension; i < ns.size(); i++)
      ns[i] = shape[i + 1];
    return Tensor<T, n - 1>(freduce_sum(node, dimension), ns);
  }
  /** Reduces one dimension of the tensor by multiplicative folding e.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0, 1, 32}, {2, 3, 4}}, {{4, 5, -6}, {6, 7, -1}}};
   * std::cout << (a.reduce_mul(0))() << std::endl;
   * // Tensor<INT32, shape: [2, 3]>(
   * // [[0, 5, -192],
   * //  [12, 21, -4]])
   * std::cout << (a.reduce_mul(1))() << std::endl;
   * // Tensor<INT32, shape: [2, 3]>(
   * // [[0, 3, 128],
   * //  [24, 35, 6]])
   * std::cout << (a.reduce_mul(2))() << std::endl;
   * // Tensor<INT32, shape: [2, 2]>(
   * // [[0, 24],
   * //  [-120, -42]])
   * }
   *
   * The results of this Tensor must be available, to
   * ensure that the method may execute the Tensor. */
  Tensor<T, n - 1> reduce_mul(int dimension) {
    std::array<size_t, n - 1> ns;
    if (dimension < 0)
      dimension = shape.size() + dimension;
    for (int i = 0; i < dimension; i++)
      ns[i] = shape[i];
    for (size_t i = dimension; i < ns.size(); i++)
      ns[i] = shape[i + 1];
    return Tensor<T, n - 1>(freduce_mul(node, dimension), ns);
  }
  /** Reduces all dimension of the tensor by multiplicatione.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0, 1, 32}, {2, 3, 4}}, {{4, 5, -6}, {6, 7, -1}}};
   * std::cout << e.reduce_sum()() << std::endl;
   * // Tensor<INT32, shape: [1]>([0])
   * }
   *
   * The results of this Tensor must be available, to
   * ensure that the method may execute the Tensor. */
  Tensor<T, 1> reduce_mul() { return flattened().reduce_mul(); }
  /** Reduces all dimension of the tensor by summation e.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0, 1, 32}, {2, 3, 4}}, {{4, 5, -6}, {6, 7, -1}}};
   * std::cout << e.reduce_sum()() << std::endl;
   * // Tensor<INT32, shape: [1]>([57])
   * }
   *
   * The results of this Tensor must be available, to
   * ensure that the method may execute the Tensor. */
  Tensor<T, 1> reduce_sum() { return flattened().reduce_sum(); }
  /** Reduces one dimension of the tensor by keeping the minimum e.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0, 1, 32}, {2, 3, 4}}, {{4, 5, -6}, {6, 7, -1}}};
   * std::cout << e.reduce_min(0)() << std::endl;
   * // Tensor<INT32, shape: [2, 3]>(
   * // [[0, 1, -6],
   * //  [2, 3, -1]])
   * std::cout << e.reduce_min(1)() << std::endl;
   * // Tensor<INT32, shape: [2, 3]>(
   * // [[0, 1, 4],
   * // [4, 5, -6]])
   * std::cout << e.reduce_min(2)() << std::endl;
   * // Tensor<INT32, shape: [2, 2]>(
   * // [[0, 2],
   * //  [-6, -1]])
   * }
   *
   * The results of this Tensor must be available, to
   * ensure that the method may execute the Tensor. */
  Tensor<T, n - 1> reduce_min(int dimension) {
    std::array<size_t, n - 1> ns;
    if (dimension < 0)
      dimension = shape.size() + dimension;
    for (int i = 0; i < dimension; i++)
      ns[i] = shape[i];
    for (size_t i = dimension; i < ns.size(); i++)
      ns[i] = shape[i + 1];
    return Tensor<T, n - 1>(freduce_min(node, dimension), ns);
  }
  /** Reduces all dimension of the tensor by keeping the maximum value e.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0, 1, 32}, {2, 3, 4}}, {{4, 5, -6}, {6, 7, -1}}};
   * std::cout << e.reduce_min()() << std::endl;
   * // Tensor<INT32, shape: [1]>([-6])
   * }
   *
   * The results of this Tensor must be available, to
   * ensure that the method may execute the Tensor. */
  Tensor<T, 1> reduce_min() { return flattened().reduce_min(); }
  /** Reduces one dimension of the tensor by keeping the maximum e.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0, 1, 32}, {2, 3, 4}}, {{4, 5, -6}, {6, 7, -1}}};
   * std::cout << e.reduce_max(0)() << std::endl;
   * // Tensor<INT32, shape: [2, 3]>(
   * // [[4, 5, 32],
   * //  [6, 7, 4]])
   * std::cout << e.reduce_max(1)() << std::endl;
   * // Tensor<INT32, shape: [2, 3]>(
   * // [[2, 3, 32],
   * //  [6, 7, -1]])
   * std::cout << e.reduce_max(2)() << std::endl;
   * // Tensor<INT32, shape: [2, 2]>(
   * // [[32, 4],
   * //  [5, 7]])
   * }
   *
   * The results of this Tensor must be available, to
   * ensure that the method may execute the Tensor. */
  Tensor<T, n - 1> reduce_max(int dimension) {
    std::array<size_t, n - 1> ns;
    if (dimension < 0)
      dimension = shape.size() + dimension;
    for (int i = 0; i < dimension; i++)
      ns[i] = shape[i];
    for (size_t i = dimension; i < ns.size(); i++)
      ns[i] = shape[i + 1];
    return Tensor<T, n - 1>(freduce_max(node, dimension), ns);
  }
  /** Reduces all dimension of the tensor by keeping the maximum value e.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0, 1, 32}, {2, 3, 4}}, {{4, 5, -6}, {6, 7, -1}}};
   * std::cout << e.reduce_max()() << std::endl;
   * // Tensor<INT32, shape: [1]>([32])
   * }
   *
   * The results of this Tensor must be available, to
   * ensure that the method may execute the Tensor. */
  Tensor<T, 1> reduce_max() { return flattened().reduce_max(); }
  /**
   * Takes the elementwise absolute value of this Tensor (negative signs are
   * removed).
   */
  Tensor<T, n> abs() const { return Tensor<T, n>(fabs_g(node), shape); }

  /** Selects a slice of the tensor with a dimension wise start index, end index
   * and step size. The arguments of this function are objects of the type
   * `TensorRange`, there may be as many arguments as dimensions or less. The
   * arguments start by the first one describing the first dimension, the second
   * one describing the second and so on. If there are less arguments than
   * dimensions, all elements of the missing last dimensions will be selected.
   *
   * Each `TensorRange` contains a `start`, `end` and `step` member.
   * `start` and `end` may be negative values, which are then subtracted from
   * the end of the tensor (e.g. `-1` means the element before last element).
   * `start` is inclusive and describes the start index of the selection per
   * dimension and `end` describes the end index per dimension and is exclusive.
   * `step` contains the per dimension step size (e.g. `2` meaning every second
   * element will be selected etc.) and may be negative as well, which reverses
   * the traversal order (the first elements are selected as the last ones). For
   * a negative step size, `start > end` must hold (for a positive of course
   * `end > start`) for each dimension. E.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0, 1, 32}, {2, 3, 4}}, {{4, 5, -6}, {6, 7, -1}}};
   * std::cout << (a.slice(TensorRange(0, 2), TensorRange(0, -1),
   *                       TensorRange(2, 0, -1)))()
   *           << std::endl;
   * // Tensor<INT32, shape: [2, 1, 2]>(
   * // [[[32, 1]],
   * //  [[-6, 5]]])
   * }
   *
   * To help with indexing there is the value
   * `TensorRange::MAX_SCOPE` which describes a index depending on the traversal
   * order in that dimension (i.e. the sign of step):
   * - for forward traversel it denotes in start the shape of that dimensions -
   *   1 (which is the last element start can index) and for end the shape of
   *   that dimension
   * - for backward traversal it denoted in start 0 and in end the element
   *   before 0 (this is necessary since otherwise it would not be possible to
   *   just inverse a dimension without eliminating values).
   * E.g.
   *
   * @code{
   * Tensor<int, 2> a{{0, 1, 2, 3}, {4, 5, 6, 7}};
   * std::cout << (a.slice(
   *               TensorRange(TensorRange::MAX_SCOPE,
   *                           TensorRange::MAX_SCOPE),
   *               TensorRange(TensorRange::MAX_SCOPE,
   *                           TensorRange::MAX_SCOPE, -1)))()
   *           << std::endl;
   * // Tensor<INT32, shape: [2, 4]>(
   * // [[3, 2, 1, 0],
   * //  [7, 6, 5, 4]])
   * }
   */
  template <typename... args>
  Tensor<T, n> slice(const args... dim_ranges) const {
    constexpr size_t num_ranges = sizeof...(args);
    static_assert(num_ranges <= n,
                  "A slice operation may only contain as many indexing ranges "
                  "as the tensor has dimensions!");
    std::array<TensorRange, num_ranges> ranges{
        static_cast<TensorRange>(dim_ranges)...};
    long starts[n], ends[n], steps[n];
    for (unsigned int i = 0; i < n; i++) {
      if (i < num_ranges) {
        if (ranges[i].start == TensorRange::MAX_SCOPE) {
          if (ranges[i].step > 0)
            starts[i] = 0;
          else
            starts[i] = shape[i] - 1;
        } else
          starts[i] = ranges[i].start;

        if (ranges[i].end == TensorRange::MAX_SCOPE) {
          if (ranges[i].step < 0)
            ends[i] = -shape[i] - 1;
          else
            ends[i] = shape[i];
        } else
          ends[i] = ranges[i].end;
        steps[i] = ranges[i].step;
      } else {
        starts[i] = 0;
        ends[i] = shape[i];
        steps[i] = 1;
      }
    }
    FGraphNode *nn = fslice_step(node, starts, ends, steps);
    std::array<size_t, n> new_shape;
    for (size_t i = 0; i < n; i++)
      new_shape[i] = nn->operation.shape[i];
    return Tensor<T, n>(nn, new_shape);
  }
  /**
   * Creates a new tensor of zeroes with the requested shape. The original
   * tensor is embedded at the given indices. Typically used to add padding to a
   * Tensor.
   * - `new_shape` an array describing the new shape (the number of dimensions
   *    stays the same).
   * - `indices` an array of indices per dimension where the Tensor should be
   *    placed inside of that dimension.
   *    Each entry describes the start index (meaning every other index prior in
   *    that dimension will only contain 0s).
   *    It is important, that the size in that dimension (denoted in
   *    `new_shape`) is large enough to hold the size of this Tensor +
   *    the start index.
   * E.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, -2, -1}}};
   * std::cout << (a.extend(std::array<size_t, 3>{3, 4, 4},
   *                        std::array<size_t, 3>{1, 0, 1}))()
   *           << std::endl;
   * // Tensor<INT32, shape: [3, 4, 4]>(
   * // [[[0, 0, 0, 0],
   * //   [0, 0, 0, 0],
   * //   [0, 0, 0, 0],
   * //   [0, 0, 0, 0]],
   * //  [[0, 0, 1, 2],
   * //   [0, 3, 4, 5],
   * //   [0, 0, 0, 0],
   * //   [0, 0, 0, 0]],
   * //  [[0, 6, 7, 8],
   * //   [0, 9, -2, -1],
   * //   [0, 0, 0, 0],
   * //   [0, 0, 0, 0]]])
   * }
   */
  Tensor<T, n> extend(std::array<size_t, n> new_shape,
                      std::array<size_t, n> indices) {
    return Tensor<T, n>(fextend(node, new_shape.data(), indices.data()),
                        new_shape);
  }
  /**
   * Creates a new tensor of zeroes with the requested shape. The original
   * tensor is embedded at the given indices and with a step size.
   * - `new_shape` an array describing the new shape (the number of dimensions
   *    stays the same).
   * - `indices` an array of indices per dimension where the Tensor should be
   *    placed inside of that dimension.
   *    Each entry describes the start index (meaning every other index prior in
   *    that dimension will only contain 0s).
   *    It is important, that the size in that dimension (denoted in
   *    `new_shape`) is large enough to hold the size of this Tensor +
   *    the start index.
   * - `steps` an array of step sizes per dimension. A step size of 2 means that
   *    between each value in that dimension of the original Tensor an
   *    additional 0 is placed. May be negative to inverse traversal order, in
   *    that case the index denotes the end of the traversal (still describes
   *    the index of the first value that occurs in the result tensor).
   * E.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, -2, -1}}};
   * std::cout << (a.extend(std::array<size_t, 3>{3, 4, 4},
   *                        std::array<size_t, 3>{0, 0, 1},
   *                        std::array<long, 3>{2, 3, -1}))()
   *           << std::endl;
   * // Tensor<INT32, shape: [3, 4, 4]>(
   * // [[[0, 2, 1, 0],
   * //   [0, 0, 0, 0],
   * //   [0, 0, 0, 0],
   * //   [0, 5, 4, 3]],
   * //  [[0, 0, 0, 0],
   * //   [0, 0, 0, 0],
   * //   [0, 0, 0, 0],
   * //   [0, 0, 0, 0]],
   * //  [[0, 8, 7, 6],
   * //   [0, 0, 0, 0],
   * //   [0, 0, 0, 0],
   * //   [0, -1, -2, 9]]])
   * }
   */
  Tensor<T, n> extend(std::array<size_t, n> new_shape,
                      std::array<size_t, n> indices,
                      std::array<long, n> steps) {
    return Tensor<T, n>(
        fextend_step(node, new_shape.data(), indices.data(), steps.data()),
        new_shape);
  }
  /**
   * Repeats dimensions of a tensor multiple times.
   * `repititions` is an array with the same number of entries as the tensor has
   * dimensions. If `repetitions` has in a dimension a value `x` the resulting
   * shape in that dimension is `x + 1` times larger than that of the origional
   * Tensor (because it is concatenated with itself `x` times),
   * such that `0` would result in no change of the shape of that dimensions, a
   * `1` would repeat the Tensor 1 time. E.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0, 1}, {1, 2}}, {{2, 3}, {3, 4}}};
   * std::cout << (a.repeat(std::array<int, 3>{0, 1, 2}))() << std::endl;
   * // Tensor<INT32, shape: [2, 4, 6]>(
   * // [[[0, 1, 0, 1, 0, 1],
   * //   [1, 2, 1, 2, 1, 2],
   * //   [0, 1, 0, 1, 0, 1],
   * //   [1, 2, 1, 2, 1, 2]],
   * //  [[2, 3, 2, 3, 2, 3],
   * //   [3, 4, 3, 4, 3, 4],
   * //   [2, 3, 2, 3, 2, 3],
   * //   [3, 4, 3, 4, 3, 4]]])
   * }
   */
  Tensor<T, n> repeat_array(std::array<int, n> repetitions) const {
    FGraphNode *nn = frepeat(node, repetitions.data());
    std::array<size_t, n> new_shape;
    for (size_t i = 0; i < n; i++)
      new_shape[i] = nn->operation.shape[i];
    return Tensor<T, n>(nn, new_shape);
  }
  /**
   * Repeats dimensions of a tensor multiple times.
   * This function allows one repetition argument per dimension (missing
   * dimensions will be filled with `0`s. For one of such a repetition argument
   * `x` the resulting shape in that dimension is `x + 1` times larger than that
   * of the origional Tensor (because it is concatenated with itself `x` times),
   * such that `0` would result in no change of the shape of that dimensions, a
   * `1` would repeat the Tensor 1 time. E.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0, 1}, {1, 2}}, {{2, 3}, {3, 4}}};
   * std::cout << (a.repeat(0, 1, 2))() << std::endl;
   * // Tensor<INT32, shape: [2, 4, 6]>(
   * // [[[0, 1, 0, 1, 0, 1],
   * //   [1, 2, 1, 2, 1, 2],
   * //   [0, 1, 0, 1, 0, 1],
   * //   [1, 2, 1, 2, 1, 2]],
   * //  [[2, 3, 2, 3, 2, 3],
   * //   [3, 4, 3, 4, 3, 4],
   * //   [2, 3, 2, 3, 2, 3],
   * //   [3, 4, 3, 4, 3, 4]]])
   * }
   */
  template <typename... args>
  Tensor<T, n> repeat(const args... repetitions) const {
    constexpr size_t num_repetitions = sizeof...(args);
    static_assert(num_repetitions <= n,
                  "A repetition operation may can only as many repetition "
                  "entries as there are dimensions in the tensor!");
    std::array<int, num_repetitions> repeat{static_cast<int>(repetitions)...};
    std::array<int, n> acc_repeat;
    for (int i = 0; i < num_repetitions; i++)
      acc_repeat[i] = repeat[i];
    for (int i = num_repetitions; i < n; i++)
      acc_repeat[i] = 0;
    FGraphNode *nn = frepeat(node, acc_repeat.data());
    std::array<size_t, n> new_shape;
    for (size_t i = 0; i < n; i++)
      new_shape[i] = nn->operation.shape[i];
    return Tensor<T, n>(nn, new_shape);
  }
  /**
   * Transposes this tensor along multiple dimensions
   * `transpositions` is an array with the same number of entries as the tensor
   * has dimensions, which gives the perumtation of dimensions. The tensor will
   * have a resulting shape in which the size in dimension `i` corresponds to
   * the former size in dimension `transpositions[i]`. `transpositions` may be
   * smaller than the number of dimensions of the original Tensor, in which case
   * the remaining dimensions will be fully transposed (0 with n-1, 1 with n-2,
   * ...). E.g.
   *
   * @code{
   * Tensor<int, 3> a{{{0, 1}, {1, 2}}, {{2, 3}, {3, 4}}};
   * std::cout << (a.transpose({1, 0, 2}))() << std::endl;
   * // Tensor<INT32, shape: [2, 2, 2]>(
   * // [[[0, 1],
   * //   [2, 3]],
   * //  [[1, 2],
   * //   [3, 4]]])
   * std::cout << (a.transpose())() << std::endl;
   * // Tensor<INT32, shape: [2, 2, 2]>(
   * // [[[0, 2],
   * //   [1, 3]],
   * //  [[1, 3],
   * //   [2, 4]]])
   * }
   */
  Tensor<T, n> transpose(std::initializer_list<int> transposition = {}) {
    std::array<int, n> acc_trans;
    int i = 0;
    for (auto it = transposition.begin(); it < transposition.end(); it++)
      acc_trans[i++] = *it;
    while (i < n)
      acc_trans[i++] = n - i - 1;
    std::array<size_t, n> new_shape;
    for (int j = 0; j < n; j++)
      new_shape[j] = shape[acc_trans[j]];
    FGraphNode *nn = ftranspose(node, acc_trans.data());
    return Tensor<T, n>(nn, new_shape);
  }
  /** Convolves the `n`-dimensional input tensor with a `n`-dimensional
   * filter kernel `kernel` and a per dimensional step size `steps` with size of
   * `n-1`. It is expected that the input and `kernel` have the same size in
   * their last dimension (which will be completly reduced by the convolution).
   * In all other dimensions the size of the input tensor should be larger or
   * equal to the size of `kernel`. The `kernel` will be 'slid' over the tensor
   * in each dimension, multiplying all values of `kernel` with the
   * corresponding ones in the tensor and summing them up to a single value and
   * moving the kernel further by the value given in `steps` in that
   * corresponding dimension.
   *
   * The implementation does an implicit right-padding, meaning that the kernel
   * will initially be placed so that its first element is aligned to
   * the first element of the tensor, but it will be slid till no kernel element
   * overlaps with any element of the tensor, multiplying all "overlapping"
   * elements with 0. If you want to mdofiy this behaviour you can use
   * `extend`, `slice` or similar.
   *
   * The resulting Tensor will therefor have a shape with dimensionality `n - 1`
   * and size of `resulting_shape[i] = 1 + (a->operation.shape[i] - 1) /
   * steps[i]`.
   *
   * @code{
   * Tensor<float, 3> t1{{{0, 1}, {1, 2}, {3, 4}},
   *                     {{5, 6}, {7, 8}, {9, 0}},
   *                     {{-1,-2},{-3,-4},{-5,-6}}};
   * Tensor<float, 3> k1{{{1, 1}, {2, 2}}};
   * Tensor<float, 2> r1 = t1.convolve(k1, 2, 1);
   * // Tensor<FLOAT32, shape: [2, 3]>(
   * // [[7.000000, 17.000000, 7.000000],
   * //  [-17.000000, -29.000000, -11.000000]])
   * }
   */
  template <typename K, typename... args>
  Tensor<stronger_return<K>, n - 1> convolve(const Tensor<K, n> &kernel,
                                             const args... steps) const {
    constexpr size_t num_steps = sizeof...(args);
    static_assert(num_steps <= n - 1,
                  "A convolve operation may only have n-1 number of steps (one "
                  "for each dimension except the last)!");
    std::array<unsigned int, num_steps> steps_arr_par{
        static_cast<unsigned int>(steps)...};
    std::array<unsigned int, n - 1> steps_arr;
    for (int i = 0; i < n - 1; i++)
      steps_arr[i] = i < num_steps ? steps_arr_par[i] : 1;
    FGraphNode *nc = fconvolve(node, kernel.get_graph_node(), steps_arr.data());
    std::array<size_t, n - 1> new_shape;
    std::copy_n(nc->operation.shape, (n - 1), new_shape.begin());
    return Tensor<stronger_return<K>, n - 1>(nc, new_shape);
  }
  /**
   * Slides `kernel` along the input tensor, multiplying it with the elements of
   * the tensor. For each element all multiplied values are summed up, so that
   * the result has the same shape as `kernel` (every element in the result is
   * the accumulated sum of the product of that element with all elements it was
   * slid over). `kernel` is initially placed so that the first element of the
   * tensor and the first element of `kernel` overlap. It is then moved for each
   * dimension `i` by `steps[i]` elements forward except for the last (steps
   * can have 1 dimension less then the tensor and `kernel`), just like it would
   * be by `convolve` with the difference, that everything is accumulated for
   * the kernel instead of the original node.
   *
   * The last dimension of the tensor and `kernel` should be equal, therefor it
   * has no step in that dimension since the complete kernel is multiplied in
   * that dimension. If you input less steps then `n-1` the remaining will be
   * set to 1.
   *
   * @code{
   * Tensor<float, 3> t1{{{0, 1}, {1, 2}, {3, 4}},
   *                     {{5, 6}, {7, 8}, {9, 0}},
   *                     {{-1,-2},{-3,-4},{-5,-6}}};
   * Tensor<float, 3> k1{{{1, 1}, {2, 2}}};
   * Tensor<float, 2> r1 = t1.slide(k1, 2, 2);
   * // Tensor<FLOAT32, shape: [1, 2, 2]>(
   * // [[[-3.000000, -3.000000],
   * //   [-4.000000, -4.000000]]])
   * }
   */
  template <typename K, typename... args>
  Tensor<stronger_return<K>, n> slide(const Tensor<K, n> &kernel,
                                      const args... steps) const {
    constexpr size_t num_steps = sizeof...(args);
    static_assert(num_steps < n,
                  "A slide operation may only have n-1 number of steps (one "
                  "for each dimension except the last)!");
    std::array<unsigned int, num_steps> steps_arr_par{
        static_cast<unsigned int>(steps)...};
    std::array<unsigned int, n - 1> steps_arr;
    for (int i = 0; i < n - 1; i++)
      steps_arr[i] = i < num_steps ? steps_arr_par[i] : 1;
    FGraphNode *nc = fslide(node, kernel.get_graph_node(), steps_arr.data());
    std::array<size_t, n> new_shape;
    std::copy_n(nc->operation.shape, n, new_shape.begin());
    return Tensor<stronger_return<K>, n>(nc, new_shape);
  }
  template <typename K, unsigned int k>
  Tensor<T, n> index(const Tensor<K, k> &indices) const {
    static_assert(std::is_same<K, int>() || std::is_same<K, long>(),
                  "Indices must be integer!");
    static_assert(k <= n,
                  "Indices must match the first dimensions of the Tensor!");
    FGraphNode *nc = findex(node, indices.get_graph_node());
    std::array<size_t, n> new_shape;
    std::copy_n(nc->operation.shape, n, new_shape.begin());
    return Tensor<T, n>(nc, new_shape);
  }
  template <typename K, unsigned int k>
  Tensor<T, n> index_set(const Tensor<T, n> &b,
                         const Tensor<K, k> &indices) const {
    static_assert(std::is_same<K, int>() || std::is_same<K, long>(),
                  "Indices must be integer!");
    static_assert(k <= n,
                  "Indices must match the first dimensions of the Tensor!");
    return Tensor<T, n>(
        findex_set(node, b.get_graph_node(), indices.get_graph_node()), shape);
  }
  /**
   * Creates "views" in an additional dimension of a fixed size windows. The
   * window of size `window_size` is slid in each dimension along the Tensor
   * starting from the beginning and moving `step_size` entries in each
   * dimension (the last dimension is moved first until it is fully traversed,
   * then the next dimension is moved - i.e. the tensor is traversed for the
   * windows like a nested for loop for each dimension). The windows are concatenated in a extra dimension, which becomes the first dimension of the result Tensor.
   *
   * E.g.
   *
   * @code{
   * Tensor<int, 3> a = {{{1, 2}, {3, 4}, {5, 6}, {7, 8}},
   *                      {{9, 10}, {11, 12}, {13, 14}, {15, 16}},
   *                      {{17, 18}, {19, 20}, {21, 22}, {23, 24}}};
   * Tensor<int, 4> b = a.sliding_window(std::array<size_t, 3>{2, 2, 2},
   *                                     std::array<unsigned int, 3>{1, 2, 1});
   * std::cout << b << std::endl;
   * // Tensor<INT32, shape: [4, 2, 2, 2]>(
   * // [[[[1, 2],
   * //    [3, 4]],
   * //   [[9, 10],
   * //    [11, 12]]],
   * //  [[[5, 6],
   * //    [7, 8]],
   * //   [[13, 14],
   * //    [15, 16]]],
   * //  [[[9, 10],
   * //    [11, 12]],
   * //   [[17, 18],
   * //    [19, 20]]],
   * //  [[[13, 14],
   * //    [15, 16]],
   * //   [[21, 22],
   * //    [23, 24]]]])
   * }
   *
   */
  Tensor<T, n + 1> sliding_window(std::array<size_t, n> window_size,
                                  std::array<unsigned int, n> step_size = {1}) {
    FGraphNode *nn =
        fsliding_window(node, window_size.data(), step_size.data());
    std::array<size_t, n + 1> ns;
    std::memcpy(ns.data(), nn->operation.shape, sizeof(size_t) * (n + 1));
    return Tensor<T, n + 1>(nn, ns);
  }

  /** Returns the underlying `FGraphNode` for use with the C-Frontend. It is
   * still memory managed by this Tensor instance, so be carefull about variable
   * lifetimes. */
  FGraphNode *get_graph_node() const { return node; }
  /**
   * Calculates the gradient of this Tensor to `dx`. A gradient is always a
   * Tensor of type `double`. `dx` needs to have been marked with `watch` before
   * construction of this Tensor and this Tensor must be constructed inside a
   * gradient context, either started by `fStartGradientContext` or a
   * `GradientContext` object.
   */
  template <typename K, unsigned int k>
  Tensor<double, k> gradient(const Tensor<K, k> &dx) const {
    return Tensor<double, k>(fCalculateGradient(this->node, dx.node), dx.shape);
  }
  /** Watches this node, i.e. collects information needed to calculate the
   * gradient with this node as a derivative */
  void watch() { fMarkGradientVariable(node); }
  /**
   * Removes the gradient mark (and subsequent memory overhead) for this node.
   * After a call to this method no subsequent gradient calculations with this
   * node as a derivative will be possible.
   */
  void unwatch() { fUnmarkGradientVariable(node); }

protected:
  FGraphNode *node;
  std::array<size_t, n> shape;
  size_t total_size;
  template <typename K>
  void initShape(const std::initializer_list<std::initializer_list<K>> &vec,
                 int i) {
    shape[i] = vec.size();
    if (vec.size() <= 0)
      flogging(F_ERROR, "No dimension of the Tensor may have size 0!");
    initShape(*vec.begin(), i + 1);
    total_size *= vec.size();
  }
  template <typename K>
  void initShape(const std::initializer_list<K> &vec, int i) {
    shape[i] = vec.size();
    total_size = vec.size();
  }
  template <typename K>
  void initShape(const std::vector<std::vector<K>> &vec, int i) {
    shape[i] = vec.size();
    if (vec.size() <= 0)
      flogging(F_ERROR, "No dimension of the Tensor may have size 0!");
    initShape(*vec.begin(), i + 1);
    total_size *= vec.size();
  }
  template <typename K> void initShape(const std::vector<K> &vec, int i) {
    shape[i] = vec.size();
    total_size = vec.size();
  }
  template <typename K>
  void bringIntoShape(std::vector<K> &dest, const std::vector<K> &src, int off,
                      int dim, size_t element_size) {
    for (size_t i = 0; i < shape[dim]; i++) {
      dest[i] = src[off + i];
    }
  }
  template <typename K, typename V>
  void bringIntoShape(std::vector<std::vector<K>> &dest,
                      const std::vector<V> &src, int off, int dim,
                      size_t element_size) {
    int nes = element_size / shape[dim];
    for (size_t i = 0; i < shape[dim]; i++) {
      dest[i] = std::vector<K>(shape[dim + 1]);
      bringIntoShape(dest[i], src, off + i * nes, dim + 1, nes);
    }
  }
};
