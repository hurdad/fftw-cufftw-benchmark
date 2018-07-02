#include <fftw3.h>
#include <benchmark/benchmark.h>
#include <math.h>

#define REAL 0
#define IMAG 1

void generate_signal(fftw_complex* signal, const int N) {
	int i;
	for (i = 0; i < N; ++i) {
		double theta = (double) i / (double) N * M_PI;
		signal[i][REAL] = 1.0 * cos(10.0 * theta) + 0.5 * cos(25.0 * theta);
		signal[i][IMAG] = 1.0 * sin(10.0 * theta) + 0.5 * sin(25.0 * theta);
	}
}

static void fftw(benchmark::State& state) {
	int N = state.range(0);

	for (auto _ : state) {
		fftw_complex signal[N];
		fftw_complex result[N];
		state.PauseTiming();
		generate_signal(signal, N);
		state.ResumeTiming();

		fftw_plan plan = fftw_plan_dft_1d(N, signal, result, FFTW_FORWARD,
				FFTW_ESTIMATE);
		fftw_execute(plan);
		fftw_destroy_plan(plan);
	}

	state.SetBytesProcessed(
			static_cast<int64_t>(state.iterations()) * N
					* sizeof(fftw_complex));
	state.SetComplexityN(N);
}
BENCHMARK(fftw)->RangeMultiplier(2)->Range(1<<10, 1<<17)->Complexity();
BENCHMARK_MAIN();


