#include <fftw3.h>
#include <benchmark/benchmark.h>
#include <math.h>

#define REAL 0
#define IMAG 1

void generate_signal(fftwf_complex* signal, const int N) {
	int i;
	for (i = 0; i < N; ++i) {
		double theta = (double) i / (double) N * M_PI;
		signal[i][REAL] = 1.0 * cos(10.0 * theta) + 0.5 * cos(25.0 * theta);
		signal[i][IMAG] = 1.0 * sin(10.0 * theta) + 0.5 * sin(25.0 * theta);
	}
}

static void fftwf(benchmark::State& state) {
	int N = state.range(0);
	for (auto _ : state) {
		fftwf_complex *in, *out;
		in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
		state.PauseTiming();
		generate_signal(in, N);
		state.ResumeTiming();
		out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
		fftwf_plan plan = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
		fftwf_execute(plan);
		fftwf_destroy_plan(plan);
		fftwf_free(in);
		fftwf_free(out);
	}
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
	state.SetBytesProcessed(
			static_cast<int64_t>(state.iterations()) * N
					* sizeof(fftwf_complex));
	state.SetComplexityN(N);
}
BENCHMARK(fftwf)->RangeMultiplier(2)->Range(1<<10, 1<<26)->Complexity();
BENCHMARK_MAIN();

