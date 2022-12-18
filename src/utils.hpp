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
#include <condition_variable>
#include <list>
#include <mutex>
#include <vector>

template <typename T> inline T *safe_mal(unsigned int count) {
  T *data = (T *)malloc(sizeof(T) * count);
  if (!data) {
    flog(F_ERROR,
         "Could not malloc '" + std::to_string(sizeof(T) * count) + "' bytes!");
  }
  return data;
}
template <typename T> inline std::string vectorString(std::vector<T> vec) {
  std::string res = "[";
  for (size_t i = 0; i < vec.size(); i++) {
    res += std::to_string(vec[i]);
    if (i != vec.size() - 1)
      res += ", ";
  }
  return res + "]";
}
template <typename T>
inline std::string vectorString(std::vector<std::vector<T>> &vec) {
  std::string res = "[";
  for (size_t i = 0; i < vec.size(); i++) {
    res += vectorString(vec[i]);
    if (i != vec.size() - 1)
      res += ",\n";
  }
  return res + "]";
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
inline size_t typeSize(FType t) {
  switch (t) {
  case F_INT32:
    return sizeof(int);
  case F_INT64:
    return sizeof(long);
  case F_FLOAT32:
    return sizeof(float);
  case F_FLOAT64:
    return sizeof(double);
  }
  return 1;
}
template <typename T> static constexpr FType toFlintType() {
  if (std::is_same<T, int>())
    return F_INT32;
  if (std::is_same<T, long>())
    return F_INT64;
  if (std::is_same<T, float>())
    return F_FLOAT32;
  if (std::is_same<T, double>())
    return F_FLOAT64;
}
template <typename T> class blocking_queue {
private:
  std::mutex mutex;
  std::condition_variable condition;
  std::list<T> queue;

public:
  void push_front(const T &el) {
    { // own visibility block to force destructor of lock
      std::unique_lock<std::mutex> lock(mutex);
      queue.push_front(el);
    }
    condition.notify_one();
  }
  T pop_front() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return !queue.empty(); });
    T foo = queue.front();
    queue.pop_front();
    return foo;
  }
};
#endif
