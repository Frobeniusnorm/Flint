#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP
#include <vector>
template <typename T>
inline std::vector<T> flattened(std::vector<std::vector<T>> vec) {
  using namespace std;
  vector<T> result;
  for (const vector<T> &v : vec) {
    result.insert(result.end(), v.begin(), v.end());
  }
  return result;
}

template <typename T>
inline std::vector<T> flattened(std::vector<std::vector<std::vector<T>>> vec) {
  using namespace std;
  vector<T> result;
  for (const vector<vector<T>> &v : vec) {
    vector<T> rec = flattened(v);
    result.insert(result.end(), rec.begin(), rec.end());
  }
  return result;
}

#endif
