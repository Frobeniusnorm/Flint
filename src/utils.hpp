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

#ifndef UTILS_HPP
#define UTILS_HPP
#include "../flint.h"
#include "logger.hpp"
#include <vector>
template <typename T> inline T *safe_mal(unsigned int count) {
  T *data = (T *)malloc(sizeof(T) * count);
  if (!data) {
    log(ERROR,
        "Could not malloc '" + std::to_string(sizeof(T) * count) + "' bytes!");
  }
  return data;
}
template <typename T> inline std::string vectorString(std::vector<T> vec) {
  std::string res = "[";
  for (int i = 0; i < vec.size(); i++) {
    res += std::to_string(vec[i]);
    if (i != vec.size() - 1)
      res += ", ";
  }
  return res + "]";
}
template <typename T>
inline std::string vectorString(std::vector<std::vector<T>> &vec) {
  std::string res = "[";
  for (int i = 0; i < vec.size(); i++) {
    res += vectorString(vec[i]);
    if (i != vec.size() - 1)
      res += ",\n";
  }
  return res + "]";
}
inline std::string typeString(FType t) {
  switch (t) {
  case INT32:
    return "int";
  case INT64:
    return "long";
  case FLOAT32:
    return "float";
  case FLOAT64:
    return "double";
  }
  return "";
}
inline size_t typeSize(FType t) {
  switch (t) {
  case INT32:
    return sizeof(int);
  case INT64:
    return sizeof(long);
  case FLOAT32:
    return sizeof(float);
  case FLOAT64:
    return sizeof(double);
  }
  return 1;
}
#endif
