#include <fftw3.h>
#include <benchmark/benchmark.h>
#include <math.h>
#include <chrono>
#include <string.h>

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

  // Allocate memory for the signal and result
  fftwf_complex* signal = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
  fftwf_complex * in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
  fftwf_complex * out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);

  //  Init fftw plan
  fftwf_plan plan = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

  //  Generate signal
  generate_signal(signal, N);

  for (auto _ : state) {
    //  Start iteration timer
    auto start = std::chrono::high_resolution_clock::now();

    // Copy signal into input of fft
    memcpy(in, signal, sizeof(fftwf_complex) * N);

    // Transform signal to fft (in => out)
    fftwf_execute(plan);

    //  Calculate elapsed time
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start);

    //  Set Iteration Time
    state.SetIterationTime(elapsed_seconds.count());

  }
  // Destroy fft plan
  fftwf_destroy_plan(plan);

  // Cleanup memory
  fftwf_free(in);
  fftwf_free(out);

  //  Save statistics
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * sizeof(fftwf_complex));
  state.SetComplexityN(N);
}
BENCHMARK(fftwf)->RangeMultiplier(2)->Range(1<<10, 1<<20)->Complexity()->UseManualTime();
BENCHMARK_MAIN();

