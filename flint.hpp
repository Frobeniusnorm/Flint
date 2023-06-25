/* Copyright 2022 David Schwarzbeck

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FLINT_HPP
#define FLINT_HPP

/* flint.hpp
 * This is the C++ implementation of Flint
 *
 * The core class of the C++ implementation is Tensor which has a template that
 * describes the dimensionality and type of the Tensor. All C++ functions use
 * the underlying implementations in flint.h.
 */

#include "flint.h"
// includes the template and helper classes
#include "flint_helper.hpp"
// includes the 1 dimensional implementation
#include "flint_1.hpp"
// includes the n dimensional implementation
#include "flint_n.hpp"

struct Flint {
  /**
   * Loads an image from the given path.
   * The image will be stored in floating point data and the shape will be h, w,
   * c where w is the width, h is the height and c are the chanels.
   */
  static Tensor<float, 3> load_image(std::string path) {
    FGraphNode *node = fload_image(path.c_str());
    return Tensor<float, 3>(node,
                            std::array<size_t, 3>{node->operation->shape[0],
                                                  node->operation->shape[1],
                                                  node->operation->shape[2]});
  }
  static void store_image(Tensor<float, 3> &t, std::string path,
                          FImageFormat format) {
    fstore_image(t.node, path.c_str(), format);
  }
  /** Sets the Logging Level of the Flint Backend */
  static void setLoggingLevel(FLogType level) { fSetLoggingLevel(level); }
  /**
   * Deallocates any resourced allocated by the corresponding backends and
   * allows them to shutdown their threads.
   */
  static void cleanup() { flintCleanup(); }

  template <typename K, unsigned int n>
  static Tensor<K, n> concat(const Tensor<K, n> &a, const Tensor<K, n> &b,
                             unsigned int ax) {
    FGraphNode *c = fconcat(a.get_graph_node(), b.get_graph_node(), ax);
    std::array<size_t, n> ns;
    for (int i = 0; i < n; i++)
      ns[i] = c->operation->shape[i];
    return Tensor<K, n>(c, ns);
  }
  /**
   * Creates a Tensor filled with random values in [0, 1) with the requested
   * shape in sizes.
   */
  template <typename... args>
  static Tensor<double, sizeof...(args)> random(args... sizes) {
    constexpr size_t dimensions = sizeof...(args);
    std::array<size_t, dimensions> shape{static_cast<size_t>(sizes)...};
    FGraphNode *node = frandom(shape.data(), dimensions);
    return Tensor<double, dimensions>(node, shape);
  }
  /**
   * Generates a Tensor containing the single given value in every entry.
   * The resulting Tensor will have a dimensionality of `sizeof...(args)` and a
   * shape denoted by each entry in `sizes`. e.g.
   * @code{
   * Tensor<double, 3> foo = Tensor<double, 3>::constant(3.141592, 2, 2, 2);
   * std::cout << foo << std::endl;
   * // Tensor<FLOAT64, shape: [2, 2, 2]>(
   * // [[[3.141592, 3.141592],
   * //  [3.141592, 3.141592]],
   * // [[3.141592, 3.141592],
   * //  [3.141592, 3.141592]]])
   * }
   */
  template <typename T, unsigned int n>
  static Tensor<T, n> constant(T value, std::array<size_t, n> shape) {
    FGraphNode *node = fconstant(value, shape.data(), n);
    return Tensor<T, n>(node, shape);
  }
  /**
   * Generates a Tensor containing the single given value in every entry.
   * The resulting Tensor will have a dimensionality of `sizeof...(args)` and a
   * shape denoted by each entry in `sizes`. e.g.
   * @code{
   * Tensor<double, 3> foo = Tensor<double, 3>::constant(3.141592, 2, 2, 2);
   * std::cout << foo << std::endl;
   * // Tensor<FLOAT64, shape: [2, 2, 2]>(
   * // [[[3.141592, 3.141592],
   * //  [3.141592, 3.141592]],
   * // [[3.141592, 3.141592],
   * //  [3.141592, 3.141592]]])
   * }
   */
  template <typename T, typename... args>
  static Tensor<T, sizeof...(args)> constant(T value, args... sizes) {
    std::array<size_t, sizeof...(args)> shape{static_cast<size_t>(sizes)...};
    return constant<T, sizeof...(args)>(value, shape);
  }
}; // namespace Flint
#endif
