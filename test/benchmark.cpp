#include "../flint.hpp"
#include "../flint_helper.hpp"
#include "plf_nanotimer.h"
#include <iostream>
#include <unordered_map>
using namespace plf;
using namespace std;

double matrix_multiplication() {
	nanotimer timer;
	vector<vector<float>> d1(64, vector<float>(64));
	for (int i = 0; i < 64; i++)
		for (int j = 0; j < 64; j++)
			d1[i][j] = i / 16.0 + j / 16.0;
	vector<vector<vector<float>>> d2(
		32, vector<vector<float>>(64, vector<float>(64)));
	for (int i = 0; i < 32; i++)
		for (int j = 0; j < 64; j++)
			for (int k = 0; k < 64; k++)
				d2[i][j][k] = (16 - i) / 2.0 * (64 - j) / 8.0 + j / 16.0;
	Tensor<float, 2> mat1(d1);
	Tensor<float, 3> mat2(d2);
	timer.start();
	for (int i = 0; i < 1000; i++) {
		Tensor<float, 3> res = mat2.matmul(mat1).pow(3.141592f);
		res.execute();
	}
	return timer.get_elapsed_ms();
}
double reduce_fun() {
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
	Tensor<float, 2> t1(d1);
	Tensor<float, 3> t2(d2);
	timer.start();
	for (int i = 0; i < 1000; i++) {
		Tensor<double, 1> res =
			((t2.sin().reduce_mul(0) * (t2 - t1).tan().reduce_sum(0))
				 .transpose({1, 0})
				 .log2()
				 .reduce_sum(0) /
			 1000.0)
				.reduce_mul()
				.abs()
				.sqrt();
		res.execute();
	}
	return timer.get_elapsed_ms();
}
double gradient_fun() {
	nanotimer timer;
	vector<vector<float>> d1(64, vector<float>(64));
	for (int i = 0; i < 64; i++)
		for (int j = 0; j < 64; j++)
			d1[i][j] = i / 16.0 + j / 16.0;
	vector<vector<vector<float>>> d2(
		32, vector<vector<float>>(64, vector<float>(64)));
	for (int i = 0; i < 32; i++)
		for (int j = 0; j < 64; j++)
			for (int k = 0; k < 64; k++)
				d2[i][j][k] = (16 - i) / 2.0 * (64 - j) / 8.0 + j / 16.0;
	Tensor<float, 2> t1(d1);
	Tensor<float, 3> t2(d2);
	t1.watch();
	t2.watch();
	timer.start();
	for (int i = 0; i < 100; i++) {
		GradientContext _;
		Tensor<double, 1> t3 =
			(t1.sqrt().matmul(t2).pow(3.141592) * (t1.log10()))
				.reduce_sum(1)
				.flattened()
				.slice(0, 128, 8)
				.pow(.75)
				.min(0);
		Tensor<double, 2> g1 = t3.gradient(t1);
		Tensor<double, 3> g2 = t3.gradient(t2);
		g1.execute();
		g2.execute();
	}
	return timer.get_elapsed_ms();
}
double convolve_fun() {
	nanotimer timer;
	vector<vector<vector<float>>> image(
		2048, vector<vector<float>>(2048, vector<float>(3, 0.8)));
	vector<vector<vector<float>>> filter(
		32, vector<vector<float>>(32, vector<float>(3, 0.5)));
	Tensor<float, 3> img_t(image);
	Tensor<float, 3> ker_t(filter);
	timer.start();
	for (int i = 0; i < 10; i++) {
		Tensor<float, 2> foo = img_t.convolve(ker_t, 8, 8);
		foo.execute();
	}
	return timer.get_elapsed_ms();
}
double convolve_grad_fun() {
	nanotimer timer;
	vector<vector<vector<float>>> image(
		2048, vector<vector<float>>(2048, vector<float>(3, 0.8)));
	vector<vector<vector<float>>> filter(
		32, vector<vector<float>>(32, vector<float>(3, 0.5)));
	Tensor<float, 3> img_t(image);
	Tensor<float, 3> ker_t(filter);
	ker_t.watch();
	timer.start();
	for (int i = 0; i < 10; i++) {
		fStartGradientContext();
		Tensor<float, 2> foo = img_t.convolve(ker_t, 8, 8);
		Tensor<float, 2> err = (foo - 0.7f).abs();
		err.execute();
		fStopGradientContext();
		Tensor<float, 3> grad = err.gradient(ker_t);
		grad.execute();
	}
	return timer.get_elapsed_ms();
}
void call_benchmarks(int benchmarks = FLINT_BACKEND_BOTH) {
	unordered_map<string, double (*)()> benches;
	benches.insert({"convolve_fun", convolve_fun});
	benches.insert({"convolve_grad_fun", convolve_grad_fun});
	benches.insert({"gradient_fun", gradient_fun});
	benches.insert({"matrix_multiplication", matrix_multiplication});
	benches.insert({"reduce_fun", reduce_fun});
	/////////////////////////////////////////////////
	unordered_map<string, tuple<double, double, double>> times;
	Flint::setLoggingLevel(F_INFO);
	if (benchmarks & FLINT_BACKEND_ONLY_CPU) {
		// cpu tests
		flintInit(FLINT_BACKEND_ONLY_CPU);
		for (const auto &bench : benches) {
			flogging(F_INFO, bench.first + "...");
			times.insert({bench.first, {bench.second(), 0, 0}});
			flogging(F_INFO,
					 "took " + to_string(std::get<0>(times[bench.first])));
		}
		flintCleanup();
	}
	if (benchmarks & FLINT_BACKEND_ONLY_GPU) {
		// gpu tests
		flintInit(FLINT_BACKEND_ONLY_GPU);
		for (const auto &bench : benches) {
			flogging(F_INFO, bench.first + "...");
			std::get<1>(times[bench.first]) = bench.second();
			flogging(F_INFO,
					 "took " + to_string(std::get<1>(times[bench.first])));
		}
		flintCleanup();
	}
	if ((benchmarks & FLINT_BACKEND_BOTH) == FLINT_BACKEND_BOTH ||
		benchmarks == 4) {
		// both tests
		flintInit(FLINT_BACKEND_BOTH);
		for (const auto &bench : benches) {
			flogging(F_INFO, bench.first + "...");
			std::get<2>(times[bench.first]) = bench.second();
			flogging(F_INFO,
					 "took " + to_string(std::get<2>(times[bench.first])));
		}
		flintCleanup();
	}
	std::cout
		<< "+------------------------+------------------+------------------"
		   "+------------------+"
		<< std::endl;
	std::cout
		<< "| benchmark name         | cpu time (ms)    | gpu time (ms)    "
		   "| jit both (ms)    |"
		<< std::endl;
	std::cout
		<< "+------------------------+------------------+------------------"
		   "+------------------+"
		<< std::endl;
	for (auto kv : times) {
		string name = kv.first;
		if (kv.first.size() > 22) {
			name = kv.first.substr(0, 20);
			name += "..";
		}
		string cpu_time = to_string(std::get<0>(kv.second));
		string gpu_time = to_string(std::get<1>(kv.second));
		string jit_time = to_string(std::get<2>(kv.second));

		for (string *str : {&cpu_time, &gpu_time, &name, &jit_time}) {
			size_t target = str == &name ? 22 : 16;
			if (str->size() > target)
				*str = str->substr(0, target);
			if (str->size() < target) {
				int missing = target - str->size();
				for (int i = 0; i < missing; i++)
					*str += " ";
			}
		}
		cout << "| " << name << " | " << cpu_time << " | " << gpu_time << " | "
			 << jit_time << " |" << endl;
		std::cout
			<< "+------------------------+------------------+----------------"
			   "--+------------------+"
			<< std::endl;
	}
}

int main(int argc, char **argv) {
	int backends = 0;
	bool lazy = false;
	if (argc > 1) {
		if (argc > 3)
			flogging(F_ERROR,
					 "Invalid number of command line arguments! Call this "
					 "program like this: benchmark [cpu] [gpu]");

		for (int i = 1; i < argc; i++) {
			if (strcmp(argv[i], "cpu") == 0)
				backends |= FLINT_BACKEND_ONLY_CPU;
			else if (strcmp(argv[i], "gpu") == 0) {
				backends |= FLINT_BACKEND_ONLY_GPU;
			} else if (strcmp(argv[i], "jit") == 0) {
				lazy = true;
			} else
				flogging(
					F_ERROR,
					"Invalid argument: " + std::string(argv[i]) +
						"! Call this program like this: benchmark [cpu] [gpu]");
		}
	}
	if (lazy)
		backends = 4;
	if (backends == 0)
		backends = FLINT_BACKEND_BOTH;
	call_benchmarks(backends);
}
