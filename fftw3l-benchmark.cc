#include <fftw3.h>
#include <benchmark/benchmark.h>
#include <math.h>

#define REAL 0
#define IMAG 1

void generate_signal(fftwl_complex* signal, const int N) {
	int i;
	for (i = 0; i < N; ++i) {
		double theta = (double) i / (double) N * M_PI;
		signal[i][REAL] = 1.0 * cos(10.0 * theta) + 0.5 * cos(25.0 * theta);
		signal[i][IMAG] = 1.0 * sin(10.0 * theta) + 0.5 * sin(25.0 * theta);
	}
}

static void fftwl(benchmark::State& state) {
	int N = state.range(0);
	for (auto _ : state) {
		fftwl_complex *in, *out;
		in = (fftwl_complex*) fftwl_malloc(sizeof(fftwl_complex) * N);
		state.PauseTiming();
		generate_signal(in, N);
		state.ResumeTiming();
		out = (fftwl_complex*) fftwl_malloc(sizeof(fftwl_complex) * N);
		fftwl_plan plan = fftwl_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
		fftwl_execute(plan);
		fftwl_destroy_plan(plan);
		fftwl_free(in);
		fftwl_free(out);
	}
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
	state.SetBytesProcessed(
			static_cast<int64_t>(state.iterations()) * N
					* sizeof(fftwl_complex));
	state.SetComplexityN(N);
}
BENCHMARK(fftwl)->RangeMultiplier(2)->Range(1<<10, 1<<26)->Complexity();
BENCHMARK_MAIN();

