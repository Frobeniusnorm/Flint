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
    flogging(F_ERROR, "Could not malloc '" + std::to_string(sizeof(T) * count) +
                          "' bytes!");
  }
  return data;
}
extern const char *fop_to_string[];
template <typename T>
static inline std::string vectorString(const std::vector<T> &vec,
                                       std::string indentation = "") {
  std::string res = "[";
  for (size_t i = 0; i < vec.size(); i++) {
    res += std::to_string(vec[i]);
    if (i != vec.size() - 1)
      res += ", ";
  }
  return res + "]";
}
template <typename T>
static inline std::string vectorString(const std::vector<std::vector<T>> &vec,
                                       std::string indentation = "") {
  std::string res = "[";
  for (size_t i = 0; i < vec.size(); i++) {
    res += vectorString(vec[i], indentation + " ");
    if (i != vec.size() - 1)
      res += ",\n" + indentation;
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
inline FType higherType(const FType a, const FType b) {
  FType highest = F_INT32;
  if (a == F_FLOAT64 || (b == F_FLOAT64))
    highest = F_FLOAT64;
  else if (a == F_FLOAT32 || (b == F_FLOAT32))
    highest = F_FLOAT32;
  else if (a == F_INT64 || (b == F_INT64))
    highest = F_INT64;
  return highest;
}
inline std::vector<std::vector<FType>> allTypePermutations(int num) {
  using namespace std;
  if (num == 1)
    return vector<vector<FType>>{
        {F_INT32}, {F_FLOAT32}, {F_INT64}, {F_FLOAT64}};
  const vector<vector<FType>> rek = allTypePermutations(num - 1);
  vector<vector<FType>> res(rek.size() * 4);
  for (int i = 0; i < rek.size(); i++) {
    int j = 0;
    for (FType ex : {F_INT32, F_INT64, F_FLOAT32, F_FLOAT64}) {
      vector<FType> old = rek[i];
      old.push_back(ex);
      res[i * 4 + j++] = old;
    }
  }
  return res;
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
inline void freeAdditionalData(FGraphNode *gn) {

  switch (gn->operation->op_type) {
  case FSLICE: {
    FSlice *s = (FSlice *)gn->operation->additional_data;
    free(s->end);
    free(s->start);
    free(s->step);
    delete s;
  } break;
  case FEXTEND: {
    FExtend *s = (FExtend *)gn->operation->additional_data;
    free(s->start);
    free(s->step);
    delete s;
  } break;
  case FCONVOLVE:
  case FSLIDE:
  case FGRADIENT_CONVOLVE:
  case FTRANSPOSE:
  case FREDUCE_SUM:
  case FREDUCE_MUL:
    free(gn->operation->additional_data);
  default:
    break;
  }
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
