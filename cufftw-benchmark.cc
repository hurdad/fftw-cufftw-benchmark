#include <cufftw.h>
#include <benchmark/benchmark.h>
#include <math.h>
#include <chrono>
#include <string.h>

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

static void cu_fftw(benchmark::State& state) {
  int N = state.range(0);

  // Allocate memory for the signal and result
  fftw_complex* signal = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
  fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
  fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

  //  Init fftw plan
  fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

  //  Generate signal
  generate_signal(signal, N);

  for (auto _ : state) {
    //  Start iteration timer
    auto start = std::chrono::high_resolution_clock::now();

    // Copy signal into input of fft
    memcpy(in, signal, sizeof(fftw_complex) * N);

    // Transform signal to fft (in => out)
    fftw_execute(plan);

    //  Calculate elapsed time
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start);

    //  Set Iteration Time
    state.SetIterationTime(elapsed_seconds.count());
  }

  // Destroy fft plan
  fftw_destroy_plan(plan);

  // Cleanup memory
  fftw_free(in);
  fftw_free(out);

  //  Save statistics
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * sizeof(fftw_complex));
  state.SetComplexityN(N);
}
BENCHMARK(cu_fftw)->RangeMultiplier(2)->Range(1<<10, 1<<26)->Complexity()->UseManualTime();
BENCHMARK_MAIN();

