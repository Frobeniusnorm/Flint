/* Copyright 2022 David Schwarzbeck

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

  This file includes the implementation of the CPU backend.
*/
#include "../../flint.h"
// #include "execution.hpp"
#include "../errors.hpp"
#include "../operations/implementation.hpp"
#include "../utils.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <queue>
#include <semaphore>
#include <stdlib.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
// virtual maximum number of threads
#define MAX_PARALLELITY 4096
static bool initialized = false;
static std::vector<std::thread *> threads;

static void threadRoutine();
FErrorType flintInit_cpu() {
	if (!initialized) {
		initialized = true;
		int cores = std::thread::hardware_concurrency();
		if (!cores)
			cores = 8;
		flogging(F_INFO,
				 "Using " + std::to_string(cores) + " threads for CPU-backend");
		threads = std::vector<std::thread *>(cores);
		for (int i = 0; i < cores; i++)
			threads[i] = new std::thread(threadRoutine);
	}
	return NO_ERROR;
}
static blocking_queue<
	std::tuple<FGraphNode *, std::vector<CPUResultData>, void *, size_t, size_t,
			   std::counting_semaphore<MAX_PARALLELITY> *>>
	thread_queue;

FErrorType flintCleanup_cpu() {
	if (initialized) {
		flogging(F_DEBUG, "Sending kill signal and poisson pills");
		initialized = false;
		for (size_t i = 0; i < threads.size(); i++)
			thread_queue.push_front({nullptr, {}, nullptr, 0, 0, nullptr});
		for (std::thread *t : threads) {
			t->join();
			delete t;
		}
	}
	return NO_ERROR;
}

static void threadRoutine() {
	while (true) {
		auto [node, pred_data, result, from, to, sem] =
			thread_queue.pop_front();
		if (!node)
			break;
		OperationImplementation::implementations[node->operation.op_type]
			->execute_cpu(node, pred_data, result, from, to);
		sem->release();
	}
}
#define PARALLEL_EXECUTION_SIZE 256 // for debugging
static void chooseExecutionMethod(FGraphNode *node,
								  std::vector<CPUResultData> pred_data,
								  void *result, size_t size) {
	const auto start = std::chrono::high_resolution_clock::now();
	const size_t score =
		size * OperationImplementation::implementations[node->operation.op_type]
				   ->operation_score(node);
	if (score >= PARALLEL_EXECUTION_SIZE && size >= threads.size()) {
		const size_t exeUnits = std::min(size, threads.size());
		const size_t workSize = size / exeUnits;
		std::counting_semaphore<MAX_PARALLELITY> *sem =
			new std::counting_semaphore<MAX_PARALLELITY>(0);
		for (size_t i = 0; i < exeUnits; i++) {
			const size_t to = i == exeUnits - 1 ? size : (i + 1) * workSize;
			thread_queue.push_front({node, pred_data, result, i * workSize,
									 to - i * workSize, sem});
		}
		for (size_t i = 0; i < exeUnits; i++)
			sem->acquire();
		delete sem;
	} else {
		OperationImplementation::implementations[node->operation.op_type]
			->execute_cpu(node, pred_data, result, 0, size);
	}
	std::chrono::duration<double, std::milli> elapsed =
		std::chrono::high_resolution_clock::now() - start;
	flogging(F_DEBUG,
			 (score >= PARALLEL_EXECUTION_SIZE && size >= threads.size()
				  ? std::string("Parallel Execution on CPU (score: " +
								std::to_string(score) + ")")
				  : std::string("Sequential Execution on CPU (score: " +
								std::to_string(score) + ")")) +
				 " took " + std::to_string(elapsed.count()) + "ms");
}
FGraphNode *fExecuteGraph_cpu_eagerly(FGraphNode *node) {
	if (!initialized)
		flintInit_cpu();
	if (node->result_data)
		return node;
	bool is_data_node = node->operation.op_type == FSTORE;
	std::vector<CPUResultData> pred_data(node->num_predecessor);
	size_t total = 1;
	for (int i = 0; i < node->operation.dimensions; i++)
		total *= node->operation.shape[i];
	void *data = nullptr;

	if (!is_data_node) {
		const std::vector<bool> reusage =
			OperationImplementation::implementations[node->operation.op_type]
				->reuse_parameter_result(node);
		// build predecessor data
		for (int i = 0; i < node->num_predecessor; i++) {
			FGraphNode *pred = node->predecessors[i];
			if (pred->result_data) {
				if (!pred->result_data->data)
					fSyncMemory(pred);
				pred_data[i].data = pred->result_data->data;
				pred_data[i].num_entries = pred->result_data->num_entries;
			} else if (pred->operation.op_type == FSTORE) {
				FStore *store = (FStore *)pred->operation.additional_data;
				pred_data[i].data = store->data;
				pred_data[i].num_entries = store->num_entries;
			} else {
				setErrorType(INTERNAL_ERROR);
				flogging(F_ERROR, "unexecuted node!");
				return nullptr;
			}
			pred_data[i].type = pred->operation.data_type;
			pred_data[i].shape = std::vector<size_t>(
				pred->operation.shape,
				pred->operation.shape + pred->operation.dimensions);
			if (!data && pred->reference_counter == 1 && !reusage.empty() &&
				reusage[i] && pred->operation.op_type != FSTORE &&
				pred != node) {
				if (pred->result_data) {
					FResultData *data = pred->result_data;
					if (data->mem_id)
						clReleaseMemObject(data->mem_id);
					delete data;
					pred->result_data = nullptr;
				}
				pred_data[i].multi_use = true;
				data = pred_data[i].data;
			}
		}
		if (!data) {
			switch (node->operation.data_type) {
			case F_INT32:
				data = safe_mal<int>(total);
				if (!data)
					return nullptr;
				break;
			case F_INT64:
				data = safe_mal<long>(total);
				if (!data)
					return nullptr;
				break;
			case F_FLOAT32:
				data = safe_mal<float>(total);
				if (!data)
					return nullptr;
				break;
			case F_FLOAT64:
				data = safe_mal<double>(total);
				if (!data)
					return nullptr;
				break;
			}
		}
		chooseExecutionMethod(node, pred_data, (double *)data, total);
	} else {
		data = ((FStore *)node->operation.additional_data)->data;
	}
	FResultData *rd = new FResultData();
	rd->data = data;
	rd->num_entries = total;
	rd->mem_id = nullptr;
	node->result_data = rd;
	return node;
}

FGraphNode *fExecuteGraph_cpu(FGraphNode *node) {
	if (!initialized)
		flintInit_cpu();
	if (node->result_data)
		return node;
	if (node->operation.op_type == FSTORE) {
		node->result_data = new FResultData();
		FStore *store = (FStore *)node->operation.additional_data;
		node->result_data->num_entries = store->num_entries;
		node->result_data->mem_id = store->mem_id;
		node->result_data->data = store->data;
		return node;
	}
	using namespace std;
	unordered_map<FGraphNode *, CPUResultData> results;
	list<FGraphNode *> toExecute; // in top down order
	{
		list<FGraphNode *> workList; // traverse bottom up
		unordered_set<FGraphNode *> inExecuteList;
		workList.push_front(node);
		const bool is_gpu_backend =
			flintInitializedBackends() & FLINT_BACKEND_ONLY_GPU;
		// collect nodes
		while (!workList.empty()) {
			FGraphNode *curr = workList.front();
			workList.pop_front();
			if (inExecuteList.find(curr) != inExecuteList.end())
				toExecute.remove(curr);
			inExecuteList.insert(curr);
			toExecute.push_front(curr);
			for (int i = 0; i < curr->num_predecessor; i++) {
				// execute on GPU if it makes more sense
				if (is_gpu_backend) {
					FGraphNode *p = curr->predecessors[i];
					const size_t score = computeScore(p, true);
					if (score >= 1024) {
						if (inExecuteList.find(p) != inExecuteList.end())
							toExecute.remove(p);
						fSyncMemory(fExecuteGraph_gpu(p));
						toExecute.push_front(p);
						inExecuteList.insert(p);
						continue;
					}
				}
				workList.push_back(curr->predecessors[i]);
			}
		}
	}
	// work them in correct oder
	for (FGraphNode *curr : toExecute) {
		// collect predecessor results
		// calculate total size
		size_t size = 1;
		for (int j = 0; j < curr->operation.dimensions; j++)
			size *= curr->operation.shape[j];
		if (size == 0)
			flogging(F_ERROR, "Illegal number of entries!");
		if (curr->operation.op_type == FSTORE || curr->result_data) {
			CPUResultData foo;
			foo.shape = vector<size_t>(curr->operation.shape,
									   curr->operation.shape +
										   curr->operation.dimensions);
			foo.type = curr->operation.data_type;
			if (curr->result_data) {
				FResultData *rd = curr->result_data;
				if (!rd->data)
					fSyncMemory(curr);
				foo.num_entries = rd->num_entries;
				if (foo.num_entries != size)
					flogging(F_ERROR, "Wrong number of entries!");
				foo.data = rd->data;
			} else {
				FStore *store = (FStore *)curr->operation.additional_data;
				foo.num_entries = store->num_entries;
				if (foo.num_entries != size)
					flogging(F_ERROR, "Wrong number of entries!");
				foo.data = store->data;
			}
			results.insert({curr, foo});
		} else {
			vector<CPUResultData> predData(curr->num_predecessor);
			void *data_to_recycle = nullptr;
			const vector<bool> reusage =
				OperationImplementation::implementations[curr->operation
															 .op_type]
					->reuse_parameter_result(curr);
			for (int i = 0; i < curr->num_predecessor; i++) {
				FGraphNode *pred = curr->predecessors[i];
				predData[i] = results[pred];
				// recycle result data of that parent if it is no longer used
				// elsewhere // TODO outside of gradient context this should be
				// possible for store too
				if (!data_to_recycle && pred->reference_counter == 1 &&
					!reusage.empty() && reusage[i] &&
					pred->operation.op_type != FSTORE && pred != node) {
					if (pred->result_data) {
						FResultData *data = pred->result_data;
						if (data->mem_id)
							clReleaseMemObject(data->mem_id);
						delete data;
						pred->result_data = nullptr;
					}
					results[pred].multi_use = true;
					predData[i].multi_use = true;
					data_to_recycle = predData[i].data;
				}
			}
			// allocate result data and execute // TODO refactor
			void *result = data_to_recycle;
			if (!result) {
				switch (curr->operation.data_type) {
				case F_INT32: {
					result = safe_mal<int>(size);
				} break;
				case F_INT64: {
					result = safe_mal<long>(size);
				} break;
				case F_FLOAT32: {
					result = safe_mal<float>(size);
				} break;
				case F_FLOAT64: {
					result = safe_mal<double>(size);
				} break;
				}
				if (!result)
					return nullptr;
			}
			chooseExecutionMethod(curr, predData, result, size);
			results.insert(
				{curr,
				 {.data = result,
				  .type = curr->operation.data_type,
				  .num_entries = size,
				  .shape = vector<size_t>(curr->operation.shape,
										  curr->operation.shape +
											  curr->operation.dimensions)}});
		}
	}
	CPUResultData final = results[node];
	if (!fIsEagerExecution()) {
		// free all other data
		for (auto &[gn, rd] : results) {
			if (gn != node && gn->operation.op_type != FSTORE &&
				!gn->result_data && gn->operation.op_type != FRESHAPE &&
				!rd.multi_use)
				free(rd.data);
		}
	} else {
		// construt a result for each node
		for (auto &[gn, rd] : results) {
			if (gn != node && gn->operation.op_type != FSTORE &&
				!gn->result_data && !rd.multi_use) {
				FResultData *result = new FResultData();
				result->data = rd.data;
				result->num_entries = rd.num_entries;
				result->mem_id = nullptr;
				gn->result_data = result;
			}
		}
	}
	// return data
	FResultData *rd = new FResultData();
	rd->data = final.data;
	rd->num_entries = final.num_entries;
	rd->mem_id = nullptr;
	node->result_data = rd;
	return node;
}
