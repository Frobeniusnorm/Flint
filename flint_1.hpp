#include "flint.h"
#include "flint_helper.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <sys/types.h>
#include <tuple>
#include <vector>
/**
 * The 1 dimensional implementation of `Tensor`.
 */
template <typename T> struct Tensor<T, 1> {
  template <typename K, unsigned int k> friend struct Tensor;
  typedef std::vector<T> storage_type;
  typedef std::initializer_list<T> init_type;
  /**
   * Creates a Tensor from a `std::vector`.
   * (`storage_type` is a recursive defined type definition, for `n=1` it is
   * just an alias for `std::vector`). E.g.
   *
   * @code{
   * Tensor<float, 1> t1{-1., 0., 1., 2.};
   * }
   */
  Tensor(storage_type data) : shape{data.size()} {
    isTensorType<T>();
    node =
        fCreateGraph(data.data(), data.size(), toFlintType<T>(), &shape[0], 1);
    node->reference_counter = 1;
  }
  /**
   * Creates a Tensor from a `std::initializer_list`.
   * (`init_type` is a recursive defined type definition, for `n=1` it is just
   * an alias for `std::initializer_list`). E.g.
   *
   * @code{
   * Tensor<float, 1> t(std::vector<float>{-1., 0., 1., 2.});
   * }
   */
  Tensor(init_type data) : shape{data.size()} {
    isTensorType<T>();

    node = fCreateGraph(std::begin(data), data.size(), toFlintType<T>(),
                        &shape[0], 1);
    node->reference_counter = 1;
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
  void operator=(const Tensor<T, 1> &other) {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
    shape = other.shape;
    node = fCopyGraph(other.node);
    node->reference_counter++;
  }
  /*
   * Move constructor. Moves every important field from `other` to this Tensor.
   * `other` is invalidated after this operation.
   */
  Tensor(Tensor &&other) {
    shape = other.shape;
    node = other.node;
    other.node = nullptr;
  }
  /**
   * Move operator. Moves every important field from `other` to this Tensor.
   * `other` is invalidated after this operation. If there was any previous
   * allocated operation node allocated by this Tensor it is cleaned up.
   */
  void operator=(Tensor &&other) {
    if (node) {
      node->reference_counter--;
      fFreeGraph(node);
    }
    shape = other.shape;
    node = other.node;
    other.node = nullptr;
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
   * Indexes the Tensor and returns the element.
   * If the underlying data is not yet computed, executes this Tensor.
   */
  T &operator[](const size_t index) {
    if (!node->result_data)
      execute();
    if (!node->result_data->data)
      fSyncMemory(node);
    return ((T *)node->result_data->data)[index];
  }
  /**
   * Generates a Tensor containing the single given value in every entry.
   * The resulting Tensor will have a dimensionality of 1 and a
   * size denoted by `size`. e.g.
   *
   * @code{
   * Tensor<double, 1> foo = Tensor<double, 1>::constant(3.141592, 3);
   * std::cout << foo << std::endl;
   * // Tensor<FLOAT64, shape: 3>([3.141592, 3.141592, 3.141592])
   * }
   */
  static Tensor<T, 1> constant(T value, size_t size) {
    FGraphNode *node = fconstant(value, &size, 1);
    return Tensor(node, size);
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
  static Tensor<T, 1> deserialize(char *data) {
    FGraphNode *node = fdeserialize(data);
    if (1 != node->operation->dimensions)
      flogging(F_ERROR, "Deserializing data of a " +
                            std::to_string(node->operation->dimensions) +
                            " dimensional Tensor into a 1 dimensional"
                            " Tensor is not possible!");
    if (toFlintType<T>() != node->operation->data_type)
      flogging(F_ERROR,
               "Deserializing data of a " +
                   FLINT_HPP_HELPER::typeString(node->operation->data_type) +
                   " Tensor into a " +
                   FLINT_HPP_HELPER::typeString(toFlintType<T>()) +
                   " Tensor is not possible!");
    return Tensor<T, 1>(node, node->operation->shape[0]);
  }
  /**
   * Deserializes the binary representation of Tensor data back to a Tensor
   * object.
   */
  static Tensor<T, 1> deserialize(std::vector<char> data) {
    return deserialize(data.data());
  }
  /** Reduces one dimension of the tensor by additive folding e.g.
   *
   * @code{
   * Tensor<int, 1> a{0, 1, 2, 3, 4, 5, 6};
   * std::cout << (a.reduce_sum())() << std::endl;
   * // Tensor<INT32, shape: 1>([21])
   * }
   *
   * The results of this Tensor must be available, to
   * ensure that the method may execute the Tensor. */
  Tensor<T, 1> reduce_sum() { return Tensor<T, 1>(freduce_sum(node, 0), 1); }
  /** Reduces one dimension of the tensor by multiplicative folding e.g.
   *
   * @code{
   * Tensor<int, 1> a{1, 2, 3, 4};
   * std::cout << (a.reduce_mul())() << std::endl;
   * // Tensor<INT32, shape: 1>([24])
   * }
   *
   * The results of this Tensor must be available, to
   * ensure that the method may execute the Tensor. */
  Tensor<T, 1> reduce_mul() { return Tensor<T, 1>(freduce_mul(node, 0), 1); }
  /** Returns the number of entries in this Tensor */
  const size_t get_shape() const { return shape[0]; }
  /**
   * Retrieves the data of the current node and converts it into a vector.
   * Executes the node if necessary (if it was not executed prior). This
   * operation has to duplicate the complete data. Since that is a memory heavy
   * and slow operation, it is recommended to use the index operator
   * `operator[]` whenever possible instead. E.g.
   *
   * @code{
   * Tensor<int, 1> foo = Tensor<int, 1>::constant(42, 5);
   * std::vector<int> foo_res = *foo;
   * // foo_res = {42, 42, 42, 42, 42}
   * }
   */
  std::vector<T> operator*() {
    if (!node->result_data)
      execute();
    if (!node->result_data->data)
      fSyncMemory(node);
    return std::vector<T>((T *)node->result_data->data,
                          (T *)node->result_data->data +
                              node->result_data->num_entries);
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
      node = fExecuteGraph(node);
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
      node = fExecuteGraph_cpu(node);
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
      node = fExecuteGraph_gpu(node);
    }
  }
  /**
   * Convenience Method that calls `execute` and returns the Tensor object
   * (the same, no new node is created!).
   */
  Tensor<T, 1> &operator()() {
    execute();
    return *this;
  }
  /**
   * Negates the elements of this Tensor.
   * E.g.
   *
   * @code{
   * Tensor<float, 1> foo = {-3, 3.141592, 42.0798, -4.3};
   * std::cout << (-foo)() << std::endl;
   * // Tensor<FLOAT32, shape: 4>([3.000000, -3.141592, -42.079800, 4.300000])
   * }
   */
  Tensor<T, 1> operator-() const { return Tensor<T, 1>(fneg(node), shape); }
  /**
   * Returns a tensor `x` with the shape of a with `x[i] = 1` if `a[i] >= 0`
   * else `x[i] = -1`. If you need to distinguish additionally for 0 values,
   * take a look at `equal`. E.g.
   *
   * @code{
   * Tensor<float, 1> foo = {-3, 3.141592, 42.0798, -4.3};
   * std::cout << (foo.sign())() << std::endl;
   * // Tensor<INT32, shape: 4>([-1, 1, 1, -1])
   * }
   */
  Tensor<int, 1> sign() const { return Tensor<int, 1>(fsign(node), shape); }
  /**
   * Returns a int tensor `x` with the shape of `this` with `x[i] = 1` if
   * `this[i] % 2 = 0` else `x[i] = 0`. This Tensor needs to have a integer
   * type. E.g.
   *
   * @code{
   * Tensor<int, 1> foo = {2, 3, 42, 7};
   * std::cout << (foo.even())() << std::endl;
   * // Tensor<INT32, shape: 4>([1, 0, 1, 0])
   * }
   */
  Tensor<int, 1> even() const {
    static_assert(std::is_same<T, int>() || std::is_same<T, long>());
    return Tensor<int, 1>(feven(node), shape);
  }
  /**
   * Converts this Tensor to a string representation.
   * If the Tensor was not yet executed, it won't be, instead of the data it
   * will say "<not yet executed>".
   */
  operator std::string() {
    FOperation *op = node->operation;
    std::string foo = "Tensor<" +
                      (op->data_type == F_INT32     ? std::string("INT32")
                       : op->data_type == F_INT64   ? std::string("INT64")
                       : op->data_type == F_FLOAT32 ? std::string("FLOAT32")
                                                    : std::string("FLOAT64")) +
                      ", shape: " + std::to_string(shape[0]) + ">(";
    if (op->op_type != FSTORE && !node->result_data)
      foo += "<not yet executed>";
    else {
      if (node->result_data) {
        FResultData *store = node->result_data;
        foo += FLINT_HPP_HELPER::vectorString(std::vector<T>(
            (T *)store->data, (T *)store->data + store->num_entries));
      } else {
        switch (op->op_type) {
        case FSTORE: {
          FStore *store = (FStore *)node->operation->additional_data;
          foo += FLINT_HPP_HELPER::vectorString(std::vector<T>(
              (T *)store->data, (T *)store->data + store->num_entries));
          break;
        }
        default: // to make the compiler shut up
          break;
        }
      }
    }
    foo += ")";
    return foo;
  }
  /**
   * Calls `deserialize` on this Tensor and pipes the returned data to the
   * stream.
   */
  friend std::ofstream &operator<<(std::ofstream &os, Tensor<T, 1> &t) {
    for (char c : t.serialize()) {
      os.put(c);
    }
    return os;
  }
  /**
   * Calls `std::string()` on this Tensor and pipes the returned string to the
   * pipe.
   */
  friend std::ostream &operator<<(std::ostream &os, Tensor<T, 1> &t) {
    os << (std::string)t;
    return os;
  }
  // to calculate the return type of two tensors at compile time
  template <typename K>
  using stronger_return =
      typename std::conditional<isStronger<K, T>(), K, T>::type;
  // OPERATIONS
  /**
   * Elementwise addition of this Tensor and `other`.
   * The datatype of the result is the datatype with
   * higher precedence.
   * */
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k> operator+(const Tensor<K, k> &other) const {
    return Tensor<stronger_return<K>, k>(fadd(node, other.node), other.shape);
  }
  /**
   * Elementwise addition of the constant `other` to this Tensor.
   * If the datatype of `K` is stronger (stronger precedence) than the datatype
   * of this Tensor `T`, `K` will be the result type, else `T`.
   */
  template <typename K>
  Tensor<stronger_return<K>, 1> operator+(const K con) const {
    return Tensor<stronger_return<K>, 1>(fadd(node, con), shape);
  }
  /**
   * Elementwise substraction of this Tensor and `other`.
   * The datatype of the result is the datatype with
   * higher precedence.
   * */
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k> operator-(const Tensor<K, k> &other) const {
    return Tensor<stronger_return<K>, k>(fsub(node, other.node), other.shape);
  }
  /**
   * Elementwise substraction of this Tensor and the constant `other`.
   * If the datatype of `K` is stronger (stronger precedence) than the datatype
   * of this Tensor `T`, `K` will be the result type, else `T`.
   */
  template <typename K>
  Tensor<stronger_return<K>, 1> operator-(const K con) const {
    return Tensor<stronger_return<K>, 1>(fsub(node, con), shape);
  }
  /**
   * Elementwise multiplication of this Tensor and `other`.
   * The datatype of the result is the datatype with
   * higher precedence.
   * */
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k> operator*(const Tensor<K, k> &other) const {
    return Tensor<stronger_return<K>, k>(fmul(node, other.node), other.shape);
  }
  /**
   * Elementwise multiplication of this Tensor and the constant `other`.
   * If the datatype of `K` is stronger (stronger precedence) than the datatype
   * of this Tensor `T`, `K` will be the result type, else `T`.
   */
  template <typename K>
  Tensor<stronger_return<K>, 1> operator*(const K con) const {
    return Tensor<stronger_return<K>, 1>(fmul(node, con), shape);
  }
  /**
   * Elementwise division of this Tensor and `other`.
   * The datatype of the result is the datatype with
   * higher precedence.
   * */
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k> operator/(const Tensor<K, k> &other) const {
    return Tensor<stronger_return<K>, k>(fdiv(node, other.node), other.shape);
  }
  /**
   * Elementwise division of this Tensor and the constant `other`.
   * If the datatype of `K` is stronger (stronger precedence) than the datatype
   * of this Tensor `T`, `K` will be the result type, else `T`.
   */
  template <typename K>
  Tensor<stronger_return<K>, 1> operator/(const K con) const {
    return Tensor<stronger_return<K>, 1>(fdiv(node, con), shape);
  }
  /**
   * Takes the elementwise power of this Tensor to `other`.
   * The datatype of the result is the datatype with
   * higher precedence.
   * */
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k> pow(const Tensor<K, k> &other) const {
    return Tensor<stronger_return<K>, k>(fpow(node, other.node), other.shape);
  }
  /**
   * Takes the elementwise power of this Tensor to the constant `other`.
   * If the datatype of `K` is stronger (stronger precedence) than the datatype
   * of this Tensor `T`, `K` will be the result type, else `T`.
   */
  template <typename K> Tensor<stronger_return<K>, 1> pow(const K other) const {
    return Tensor<stronger_return<K>, 1>(fpow(node, other), shape);
  }
  /**
   * Elementwise minimum of this Tensor and `other`. Per element either the
   * entry of this Tensor or `other` is returned, depending on which is smaller.
   * The datatype of the result is the datatype with
   * higher precedence.
   * */
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k> min(const Tensor<K, k> &other) const {
    return Tensor<stronger_return<K>, k>(fmin(node, other.node), other.shape);
  }
  /**
   * Elementwise minimum of this Tensor and the constant `other`. Per element
   * either the entry of this Tensor or `other` is returned, depending on which
   * is smaller. The datatype of the result is the datatype with higher
   * precedence.
   * */
  template <typename K> Tensor<stronger_return<K>, 1> min(const K other) const {
    return Tensor<stronger_return<K>, 1>(fmin(node, other), shape);
  }
  /**
   * Elementwise maximum of this Tensor and `other`. Per element either the
   * entry of this Tensor or `other` is returned, depending on which is larger.
   * The datatype of the result is the datatype with
   * higher precedence.
   * */
  template <typename K, unsigned int k>
  Tensor<stronger_return<K>, k> max(const Tensor<K, k> &other) const {
    return Tensor<stronger_return<K>, k>(fmax(node, other.node), other.shape);
  }
  /**
   * Elementwise maximum of this Tensor and the constant `other`. Per element
   * either the entry of this Tensor or `other` is returned, depending on which
   * is larger. The datatype of the result is the datatype with higher
   * precedence.
   * */
  template <typename K> Tensor<stronger_return<K>, 1> max(const K other) const {
    return Tensor<stronger_return<K>, 1>(fmax(node, other), shape);
  }
  /**
   * Takes the elementwise natural logarithm of this Tensor.
   */
  Tensor<to_float<T>, 1> log() {
    return Tensor<to_float<T>, 1>(flog(node), shape);
  }
  /**
   * Takes the elementwise logarithm dualis of this Tensor.
   */
  Tensor<to_float<T>, 1> log2() {
    return Tensor<to_float<T>, 1>(flog2(node), shape);
  }
  /**
   * Takes the elementwise logarithm to basis 10 of this Tensor.
   */
  Tensor<to_float<T>, 1> log10() {
    return Tensor<to_float<T>, 1>(flog10(node), shape);
  }
  /**
   * Takes the elementwise square root of this Tensor.
   */
  Tensor<to_float<T>, 1> sqrt() {
    return Tensor<to_float<T>, 1>(fsqrt_g(node), shape);
  }
  /**
   * Takes the elementwise sinus of this Tensor.
   */
  Tensor<to_float<T>, 1> sin() {
    return Tensor<to_float<T>, 1>(fsin(node), shape);
  }
  /**
   * Takes the elementwise cosinus of this Tensor.
   */
  Tensor<to_float<T>, 1> cos() {
    return Tensor<to_float<T>, 1>(fcos(node), shape);
  }
  /**
   * Takes the elementwise tangents of this Tensor.
   */
  Tensor<to_float<T>, 1> tan() {
    return Tensor<to_float<T>, 1>(ftan(node), shape);
  }
  /**
   * Takes the elementwise arcsinus of this Tensor (`sin^(-1)`).
   */
  Tensor<to_float<T>, 1> asin() {
    return Tensor<to_float<T>, 1>(fasin(node), shape);
  }
  /**
   * Takes the elementwise arccosinus of this Tensor (`cos^(-1)`).
   */
  Tensor<to_float<T>, 1> acos() {
    return Tensor<to_float<T>, 1>(facos(node), shape);
  }
  /**
   * Takes the elementwise arctangents of this Tensor (`tan^(-1)`).
   */
  Tensor<to_float<T>, 1> atan() {
    return Tensor<to_float<T>, 1>(fatan(node), shape);
  }
  /**
   * Converts this Tensor (and the underlying data) to type `K` given in the
   * template. `K` must be one of `int`, `long`, `float`, `double`. The data is
   * converted, not reinterpreted.
   */
  template <typename K> Tensor<K, 1> convert() const {
    return Tensor<K, 1>(fconvert(node, toFlintType<K>()), shape);
  }
  /**
   * Takes the elementwise absolute value of this Tensor (negative signs are
   * removed).
   */
  Tensor<T, 1> abs() const { return Tensor<T, 1>(fabs_g(node), shape); }
  /**
   * Compares this tensor and `other` elementwise and returns a 0,1 integer
   * Tensor. `0` denotes that `this >= other`, `1` that `this < other` for that
   * element.
   */
  template <typename K, unsigned int k>
  Tensor<int, k> operator<(const Tensor<K, k> &other) const {
    return Tensor<int, k>(fless(node, other.node), other.shape);
  }
  /**
   * Compares this tensor and the constant `other` elementwise and returns a 0,1
   * integer Tensor. `0` denotes that `this >= other`, `1` that `this < other`
   * for that element.
   */
  template <typename K> Tensor<int, 1> operator<(const K other) const {
    return Tensor<int, 1>(fless(node, other));
  }
  /**
   * Compares this tensor and `other` elementwise and returns a 0,1 integer
   * Tensor. `0` denotes that `this <= other`, `1` that `this > other` for that
   * element.
   */
  template <typename K, unsigned int k>
  Tensor<int, 1> operator>(const Tensor<K, k> &other) const {
    return Tensor<int, k>(fgreater(node, other.node), other.shape);
  }
  /**
   * Compares this tensor and the constant `other` elementwise and returns a 0,1
   * integer Tensor. `0` denotes that `this <= other`, `1` that `this > other`
   * for that element.
   */
  template <typename K> Tensor<int, 1> operator>(const K other) const {
    return Tensor<int, 1>(fgreater(node, other));
  }
  /**
   * Compares this tensor and `other` elementwise and returns a 0,1 integer
   * Tensor. `0` denotes that `this != other`, `1` that `this == other`.
   */
  template <typename K, unsigned int k>
  Tensor<int, k> equal(const Tensor<K, k> &other) const {
    return Tensor<int, k>(fequal(node, other.node), other.shape);
  }
  /**
   * Compares this tensor and the constant `other` elementwise and returns a 0,1
   * integer Tensor. `0` denotes that `this != other`, `1` that `this == other`.
   */
  template <typename K> Tensor<int, 1> equal(const K other) const {
    return Tensor<int, 1>(fequal(node, other));
  }
  /**
   * Slices a selection of the Tensor beginning by `start` (inclusive), ending
   * with `end` (exclusive) by a step size `step`. The step size may be negative
   * in which case traversal order changes, therefor `start > end` must hold
   * (for forward traversal of course `end > start`). E.g.
   *
   * @code{
   * Tensor<int, 1> a{1, 2, 3, 4, 5, 6, 7, 8};
   * std::cout << (a.slice(6, 1, -2))() << std::endl;
   * // Tensor<INT32, shape: 3>([7, 5, 3])
   * }
   *
   * To help with indexing there is the value
   * `TensorRange::MAX_SCOPE` which describes a index depending on the traversal
   * order in that dimension (i.e. the sign of step):
   * - for forward traversel it denotes in start the shape of this Tensor -
   *   1 (which is the last element start can index) and for end the shape of
   *   this Tensor.
   * - for backward traversal it denotes in start 0 and in end the element
   *   before 0 (this is necessary since otherwise it would not be possible to
   *   just inverse a dimension without eliminating values).
   */
  Tensor<T, 1> slice(long start = 0, long end = TensorRange::MAX_SCOPE,
                     long step = 1) const {
    if (start == TensorRange::MAX_SCOPE)
      start = shape[0] - 1;
    if (end == TensorRange::MAX_SCOPE)
      end = shape[0];
    FGraphNode *nn = fslice_step(node, &start, &end, &step);
    return Tensor<T, 1>(nn, nn->operation->shape[0]);
  }
  /** Returns the underlying `FGraphNode` for use with the C-Frontend. It is
   * still memory managed by this Tensor instance, so be carefull about variable
   * lifetimes. */
  FGraphNode *get_graph_node() const { return node; }
  /**
   * Repeats this Tensor `repetitions` times. A value of `0` would yield the
   * input Tensor. E.g.
   *
   * @code{
   * Tensor<int, 1> a{0, 1, -1};
   * std::cout << (a.repeat(2))() << std::endl;
   * // Tensor<INT32, shape: 7>([0, 1, -1, 0, 1, -1, 0, 1, -1])
   * }
   */
  Tensor<T, 1> repeat(int repetitions) const {
    FGraphNode *nn = frepeat(node, &repetitions);
    return Tensor<T, 1>(nn, (shape[0] * repetitions + 1));
  }
  /**
   * Calculates the gradient of this Tensor to `dx`. A gradient is always a
   * Tensor of type `double`.
   */
  template <typename K, unsigned int k>
  Tensor<double, k> gradient(const Tensor<K, k> &dx) {
    return Tensor<double, k>(fCalculateGradient(this->node, dx.node), dx.shape);
  }
  /** Watches this node, i.e. collects information needed to calculate the
   * gradient with this node as a derivative */
  void watch() { fMarkGradientVariable(node); }
  /**
   * Removes the gradient mark (ans subsequent memory overhead) for this node.
   * After a call to this method no subsequent gradient calculations with this
   * node as a derivative will be possible.
   */
  void unwatch() { fUnmarkGradientVariable(node); }

protected:
  Tensor(FGraphNode *node, size_t shape) : node(node), shape{shape} {
    node->reference_counter++;
    fOptimizeMemory(node); // should be legal here, since C++ header adjust
                           // reference_counter
  }
  Tensor(FGraphNode *node, std::array<size_t, 1> shape)
      : node(node), shape(shape) {
    node->reference_counter++;
    fOptimizeMemory(node); // should be legal here, since C++ header adjust
                           // reference_counter
  }
  FGraphNode *node;
  std::array<size_t, 1> shape;
};
