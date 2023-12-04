#ifndef TWINE
#define TWINE
#include <list>
#include <string>
/** A fast string builder */
class Twine {
  std::list<std::string> strings;
  size_t total_length = 0;
public:
  void append(std::string s) {
    strings.push_back(s);
    total_length += s.size();
  }
  void operator+=(std::string s) {
    append(s);
  }
  void prepend(std::string s) {
    strings.push_front(s);
    total_length += s.size();
  }
  std::string string() const {
    std::string res;
    res.reserve(total_length);
    for (const std::string& s : strings) {
      res += s;
    }
    return res;
  }
};
#endif
