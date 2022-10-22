#include "plf_nanotimer.h"

#include "../flint.hpp"
#include <iostream>
#include <unordered_map>
using namespace plf;
using namespace std;

double matrix_multiplication(bool backend) {
  nanotimer timer;
  vector<vector<float>> d1(64, vector<float>(64));
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 64; j++)
      d1[i][j] = i / 16.0 + j / 16.0;
  vector<vector<vector<float>>> d2(
      16, vector<vector<float>>(64, vector<float>(64)));
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 64; j++)
      for (int k = 0; k < 64; k++)
        d2[i][j][k] = (16 - i) / 2.0 * (64 - j) / 8.0 + j / 16.0;
  Tensor<float, 2> mat1(d1);
  Tensor<float, 3> mat2(d2);
  timer.start();
  for (int i = 0; i < 1000; i++) {
    Tensor<float, 3> res = mat2.matmul(mat1).pow(3.141592f);
    if (backend)
      res.execute_gpu();
    else
      res.execute_cpu();
  }
  return timer.get_elapsed_ms();
}

// 0 = cpu, 1 = gpu, 2 = both
void call_benchmarks(int benchmarks = 2) {
  unordered_map<string, pair<double, double>> times;
  flintInit(benchmarks == 0 || benchmarks == 2, benchmarks > 0);
  enable_eager_execution();
  fSetLoggingLevel(4);
  times.insert(
      {"matrix multiplication",
       pair{benchmarks == 0 || benchmarks == 2 ? matrix_multiplication(false)
                                               : 0,
            benchmarks > 0 ? matrix_multiplication(true) : 0}});
  flintCleanup();
  std::cout
      << "+------------------------+------------------+------------------+"
      << std::endl;
  std::cout
      << "| benchmark name         | cpu time (ms)    | gpu time (ms)    |"
      << std::endl;
  std::cout
      << "+------------------------+------------------+------------------+"
      << std::endl;
  for (auto kv : times) {
    string name = kv.first;
    if (kv.first.size() > 22) {
      name = kv.first.substr(0, 20);
      name += "..";
    }
    string cpu_time = to_string(kv.second.first);
    string gpu_time = to_string(kv.second.second);

    for (string *str : {&cpu_time, &gpu_time, &name}) {
      size_t target = str == &name ? 22 : 16;
      if (str->size() > target)
        *str = str->substr(0, target);
      if (str->size() < target) {
        int missing = target - str->size();
        for (int i = 0; i < missing; i++)
          *str += " ";
      }
    }
    cout << "| " << name << " | " << cpu_time << " | " << gpu_time << " |"
         << endl;
    std::cout
        << "+------------------------+------------------+------------------+"
        << std::endl;
  }
}

int main(int argc, char **argv) {
  bool cpu = true, gpu = true;
  if (argc > 1) {
    cpu = gpu = false;
    if (argc > 3)
      flog(F_ERROR, "Invalid number of command line arguments! Call this "
                    "program like this: benchmark [cpu] [gpu]");

    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "cpu") == 0)
        cpu = true;
      else if (strcmp(argv[i], "gpu") == 0)
        gpu = true;
      else
        flog(F_ERROR,
             "Invalid argument: " + std::string(argv[i]) +
                 "! Call this program like this: benchmark [cpu] [gpu]");
    }
  }
  call_benchmarks(cpu && gpu ? 2 : cpu ? 0 : 1);
}
