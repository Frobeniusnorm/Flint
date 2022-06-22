#include "logger.hpp"
template <typename T> inline T *safe_mal(unsigned int count) {
  T *data = malloc(sizeof(T) * count);
  if (!data) {
    log(ERROR,
        "Could not malloc '" + std::to_string(sizeof(T) * count) + "' bytes!");
  }
  return data;
}
