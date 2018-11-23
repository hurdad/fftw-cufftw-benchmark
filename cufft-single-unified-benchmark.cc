#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>
#include <benchmark/benchmark.h>
#include <math.h>
#include <chrono>

#define REAL 0
#define IMAG 1

void generate_signal(cufftComplex* signal, const int N) {
  int i;
  for (i = 0; i < N; ++i) {
    double theta = (double) i / (double) N * M_PI;
    signal[i].x = 1.0 * cos(10.0 * theta) + 0.5 * cos(25.0 * theta);
    signal[i].y = 1.0 * sin(10.0 * theta) + 0.5 * sin(25.0 * theta);
  }
}

static void cu_fft_single_unified(benchmark::State& state) {
  int N = state.range(0);

  // Allocate host memory for the signal
  cufftComplex *signal = (cufftComplex *) malloc(sizeof(cufftComplex) * N);
  cufftComplex *in, *out;
  checkCudaErrors(cudaMallocManaged(&in, sizeof(cufftComplex) * N));
  checkCudaErrors(cudaMallocManaged(&out, sizeof(cufftComplex) * N));

  //  Init fftw plan
  cufftHandle plan;
  checkCudaErrors(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

  //  Generate signal
  generate_signal(signal, N);

  for (auto _ : state) {
    //  Start iteration timer
    auto start = std::chrono::high_resolution_clock::now();

    // Copy signal into input of fft
    memcpy(in, signal, sizeof(cufftComplex) * N);

    // Transform signal to time
    checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)in, (cufftComplex *)out, CUFFT_FORWARD));
    cudaDeviceSynchronize();

    //  Calculate elapsed time
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start);

    //  Set Iteration Time
    state.SetIterationTime(elapsed_seconds.count());
  }

  // Destroy CUFFT context
  checkCudaErrors(cufftDestroy(plan));

  // Cleanup memory
  free(signal);
  checkCudaErrors(cudaFree(in));
  checkCudaErrors(cudaFree(out));

  //  Save statistics
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * sizeof(cufftComplex));
  state.SetComplexityN(N);
}
BENCHMARK(cu_fft_single_unified)->RangeMultiplier(2)->Range(1<<10, 1<<20)->Complexity()->UseManualTime();
BENCHMARK_MAIN();

