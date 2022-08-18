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
