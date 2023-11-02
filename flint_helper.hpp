#ifndef FLINT_HELPER_HPP
#define FLINT_HELPER_HPP
#include "flint.h"
#include <array>
#include <string>
#include <tuple>
#include <vector>
/**
 * This is the base class of the C++ implementation of Flint.
 *
 * Instances of a implementation of this template wrap around the
 * `FGraphNode` struct by providing C++ style operations and a template
 * representing the underlying datatype of the node (adding type safety) and
 * its dimensionality (sometimes refered to as rank). That allows conversion
 * to STL objects like the `operator*` does and dimensionality safety for
 * operations like `operator[]` or `slice`.
 *
 * When using it it behaves like a single Tensor representation (i.e.
 * operations can be called on it, its data may be queried), but internally
 * it may rather store applied operations and parameters for later lazy
 * execution.
 *
 * When you apply an operation to an instance it usually returns a new
 * `Tensor` object, representing that operation applied to the old object.
 * If eager execution is enabled (see `Flint::enable_eager_execution()`) the
 * operation is directly executed with the generation of the new object,
 * else it only executes if you query its data (with `operator*` or
 * `operator[]`) or if a previous operation requires its data (keep in mind
 * that some operations have to execute the operations of their parameters
 * directly, because their data is already completly needed during execution
 * e.g. reduce operations or matrix multiplication).
 *
 * The template is recursively defined on the dimensionality `n`. Meaning
 * there are two implementations: one for the basis case `n=1` and one for
 * the general case `n>1`. The interface should not differ much, except that
 * some operations that are dimension specific behave differently.
 */
template <typename T, unsigned int n> struct Tensor;
/**
 * Useful helper functions used by the library itself.
 */
namespace FLINT_HPP_HELPER {
/**
 * Transforms a vector of arbitrary recursive dimensions to a string
 */
template <typename T>
static inline std::string vectorString(const std::vector<T> &vec,
                                       std::string indentation = "") {
  std::string res = "{";
  for (size_t i = 0; i < vec.size(); i++) {
    res += std::to_string(vec[i]);
    if (i != vec.size() - 1)
      res += ", ";
  }
  return res + "}";
}
template <typename T>
static inline std::string vectorString(const std::vector<std::vector<T>> &vec,
                                       std::string indentation = "") {
  std::string res = "{";
  for (size_t i = 0; i < vec.size(); i++) {
    res += vectorString(vec[i], indentation + " ");
    if (i != vec.size() - 1)
      res += ",\n" + indentation;
  }
  return res + "}";
}
/**
 * Transforms an array of arbitrary recursive dimensions to a string
 */
template <typename T, size_t n>
static inline std::string arrayString(const std::array<T, n> &vec) {
  std::string res = "{";
  for (size_t i = 0; i < n; i++) {
    res += std::to_string(vec[i]);
    if (i != vec.size() - 1)
      res += ", ";
  }
  return res + "}";
}
template <typename T, size_t n, size_t k>
static inline std::string
arrayString(const std::array<std::array<T, k>, n> &vec) {
  std::string res = "[";
  for (size_t i = 0; i < n; i++) {
    res += arrayString(vec[i]);
    if (i != n - 1)
      res += ",\n";
  }
  return res + "]";
}
template <typename E, typename T>
static std::vector<E> flattened(const std::vector<std::vector<T>> vec) {
  using namespace std;
  vector<T> result;
  for (const vector<T> &v : vec) {
    result.insert(result.end(), v.begin(), v.end());
  }
  return result;
}

template <typename E, typename T>
static std::vector<E>
flattened(const std::vector<std::vector<std::vector<T>>> vec) {
  using namespace std;
  vector<E> result;
  for (const vector<vector<T>> &v : vec) {
    vector<E> rec = flattened<E>(v);
    result.insert(result.end(), rec.begin(), rec.end());
  }
  return result;
}
template <typename E, typename T>
static std::vector<E>
flattened(const std::initializer_list<std::initializer_list<T>> vec) {
  using namespace std;
  vector<T> result;
  for (const initializer_list<T> &v : vec) {
    result.insert(result.end(), v.begin(), v.end());
  }
  return result;
}

template <typename E, typename T>
static std::vector<E> flattened(
    const std::initializer_list<std::initializer_list<std::initializer_list<T>>>
        vec) {
  using namespace std;
  vector<E> result;
  for (const initializer_list<initializer_list<T>> &v : vec) {
    vector<E> rec = flattened<E>(v);
    result.insert(result.end(), rec.begin(), rec.end());
  }
  return result;
}
inline std::string typeString(FType t) {
  switch (t) {
  case F_INT32:
    return "int";
  case F_INT64:
    return "long";
  case F_FLOAT32:
    return "float";
  case F_FLOAT64:
    return "double";
  }
  return "";
}
}; // namespace FLINT_HPP_HELPER
/** statically checks if the given type is one of the allowed tensor types */
template <typename T> static constexpr void isTensorType() {
  static_assert(std::is_same<T, int>() || std::is_same<T, float>() ||
                    std::is_same<T, long>() || std::is_same<T, double>(),
                "Only integer and floating-point Tensor types are allowed");
}
/** checks type precedence (e.g. `isStronger<int, double>() = false,
 * isStronger<float, long>() = true`)*/
template <typename K, typename V> static constexpr bool isStronger() {
  const int a = std::is_same<K, int>()     ? 0
                : std::is_same<K, long>()  ? 1
                : std::is_same<K, float>() ? 2
                                           : 3;
  const int b = std::is_same<V, int>()     ? 0
                : std::is_same<V, long>()  ? 1
                : std::is_same<V, float>() ? 2
                                           : 3;
  return a >= b;
}
/** Transforms a C/C++ type to a `FType` */
template <typename T> static constexpr FType toFlintType() {
  if (std::is_same<T, int>())
    return F_INT32;
  if (std::is_same<T, long>())
    return F_INT64;
  if (std::is_same<T, float>())
    return F_FLOAT32;
  if (std::is_same<T, double>())
    return F_FLOAT64;
  return F_INT32;
}
template <typename K> static constexpr bool isInt() {
  return std::is_same<K, int>() || std::is_same<K, long>();
}
/** Transforms integer types to doubles (for all other types returns identity)
 */
template <typename T>
using to_float = typename std::conditional<isInt<T>(), double, T>::type;
/**
 * Encapsulates the data of a tensor. Is only valid as long as the Tensor is
 * valid. Provides an interface for index operations on multidimensional data.
 */
template <typename T, unsigned int dimensions> class TensorView;
/** One dimensional TensorView, either of a one dimensional Tensor or an already
 * indexed one. Directly accesses the result data. This TensorView is only valid
 * as long as the original Tensor (and its data) is valid.*/
template <typename T> class TensorView<T, 1> {
  T *data;
  const size_t already_indexed;
  const size_t shape;

public:
  TensorView(T *data, const std::vector<size_t> shape,
             const size_t already_indexed)
      : data(data), already_indexed(already_indexed), shape(shape[0]) {}
  /**
   * Returns a read-write-reference to the index data entry of the Tensor-data.
   * Only valid as long as the original Tensor is valid.
   */
  T &operator[](size_t index) { return data[already_indexed + index]; }
  size_t size() const { return shape; }
};
/** Multi dimensional TensorView. Indirectly indexes the data, which is only
 * accessible when as many indices as dimensions are given. This TensorView is
 * only valid as long as the original Tensor (and its data) is valid. Needed to
 * abstract multidimensional indexing. */
template <typename T, unsigned int n> class TensorView {
  T *data;
  const size_t already_indexed;
  const std::vector<size_t> shape;

public:
  TensorView(T *data, const std::vector<size_t> shape,
             const size_t already_indexed)
      : data(data), already_indexed(already_indexed), shape(shape) {}
  /**
   * Returns a new TensorView object with one more index for the current
   * dimension (i.e. the new TensorView has one dimension less). Only valid as
   * long as the original Tensor is valid.
   */
  TensorView<T, n - 1> operator[](size_t index) {
    std::vector<size_t> ns(shape.size() - 1);
    for (size_t i = 0; i < shape.size() - 1; i++) {
      ns[i] = shape[i + 1];
      index *= shape[i + 1];
    }
    return TensorView<T, n - 1>(data, ns, already_indexed + index);
  }
};
/**
 * Describes a slice operation for one dimension.
 */
struct TensorRange {
  static const long MAX_SCOPE = 2147483647;
  long start = 0;
  long end = MAX_SCOPE;
  long step = 1;
  TensorRange() = default;
  TensorRange(std::tuple<long, long, long> range_vals)
      : start(std::get<0>(range_vals)), end(std::get<1>(range_vals)),
        step(std::get<2>(range_vals)) {}
  TensorRange(std::initializer_list<long> range_vals) {
    if (range_vals.size() > 0)
      start = *range_vals.begin();
    if (range_vals.size() > 1)
      end = *(range_vals.begin() + 1);
    if (range_vals.size() > 2)
      step = *(range_vals.begin() + 2);
  }
  TensorRange(long start, long end = MAX_SCOPE, long step = 1)
      : start(start), end(end), step(step) {}
};
/**
 * Starts a gradient context on construction and stops it on destruction.
 * Because of the overhead it is advised to stop a gradient context as soon as
 * possible, so try to keep the lifetime of this object as short as possible as
 * well.
 * For all Tensors which were constructed during the lifetime of this object the
 * gradient to a watched variable may be computed. See `fStartGradientContext`
 * and `fStopGradientContext`.
 */
struct GradientContext {
  GradientContext() { fStartGradientContext(); }
  ~GradientContext() { fStopGradientContext(); }
};
/**
 * Initializes Flint on construction and cleans it up on destruction.
 * See `flintInit` and `flintCleanup`
 */
struct FlintContext {
  /** Initializes both backends */
  FlintContext() { flintInit(FLINT_BACKEND_BOTH); }
  /** Received a value of `FLINT_BACKEND_BOTH`, `FLINT_BACKEND_CPU` and
   * `FLINT_BACKEND_GPU` */
  FlintContext(int backends) { flintInit(backends); }
  ~FlintContext() { flintCleanup(); }
};
#endif
