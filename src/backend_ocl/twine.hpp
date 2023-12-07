#ifndef TWINE
#define TWINE
#include <list>
#include <string>
/** A fast string builder */
class Twine {
		std::list<std::string> strings;
		size_t total_length = 0;

	public:
		Twine() = default;
		Twine(std::string s) : strings{s} {}
		void append(std::string s) {
			strings.push_back(s);
			total_length += s.size();
		}
		void operator+=(std::string s) { append(s); }
		void prepend(std::string s) {
			strings.push_front(s);
			total_length += s.size();
		}
		void append(const Twine& t) {
			for (const std::string &s : t.strings)
				strings.push_back(s);
			total_length += t.total_length;
		}
		void prepend(const Twine& t) {
			for (auto i = t.strings.rbegin(); i != t.strings.rend(); i++)
				strings.push_front(*i);
			total_length += t.total_length;
		}
		operator std::string() const {
			std::string res;
			res.reserve(total_length);
			for (const std::string &s : strings) {
				res += s;
			}
			return res;
		}
};
#endif
