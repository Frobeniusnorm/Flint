/* Copyright 2023 David Schwarzbeck
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */
#include "implementation.hpp"
#include "binary_arithmetic.hpp"
#include "comparison.hpp"
#include "gen_data.hpp"
#include "index_modification.hpp"
#include "reductions.hpp"
#include "shape_modification.hpp"
#include "convolution.hpp"
#include "pooling.hpp"
#include "unary_arithmetic.hpp"
#include "sliding_windows.hpp"
struct NopImpl : OperationImplementation {
		void execute_cpu(const FGraphNode *node,
						 std::vector<CPUResultData> predecessor_data,
						 void *__restrict__ result, size_t from,
						 size_t size) override {}
		void generate_gpu_lazy() override {}
		void generate_gpu_eager() override {}
};

std::vector<OperationImplementation *>
	OperationImplementation::implementations = {new NopImpl(), // store
												new GenRandomImpl(),
												new GenConstantImpl(),
												new GenArangeImpl(),
												new AddImpl(),
												new SubImpl(),
												new MulImpl(),
												new DivImpl(),
												new PowImpl(),
												new NegImpl(),
												new LogImpl(),
												new SignImpl(),
												new EvenImpl(),
												new Log2Impl(),
												new Log10Impl(),
												new SinImpl(),
												new CosImpl(),
												new TanImpl(),
												new ASinImpl(),
												new ACosImpl(),
												new ATanImpl(),
												new SqrtImpl(),
												new ExpImpl(),
												new FlattenImpl(),
												new MatMulImpl(),
												new ConversionImpl(),
												new FlattenImpl(),
												new MinImpl(),
												new MaxImpl(),
												new ReduceSumImpl(),
												new ReduceMulImpl(),
												new ReduceMinImpl(),
												new ReduceMaxImpl(),
												new SliceImpl(),
												new AbsImpl(),
												new RepeatImpl(),
												new TransposeImpl(),
												new ExtendImpl(),
												new ConcatImpl(),
												new LessImpl(),
												new EqualImpl(),
												new GreaterImpl(),
												new ConvolveImpl(),
												new GradientConvolve1Impl(),
												new GradientConvolve2Impl(),
												new IndexImpl(),
												new SetIndexImpl(),
												new SlidingWindowImpl(),
												new UnslideWindowImpl(),
												new PoolingMaxImpl(),
												new PoolingSumImpl(),
												new GradientPoolingMax()};
