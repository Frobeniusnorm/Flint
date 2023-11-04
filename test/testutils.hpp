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
