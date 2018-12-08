#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>
#include <benchmark/benchmark.h>
#include <math.h>
#include <chrono>

#define REAL 0
#define IMAG 1

void generate_signal(cuDoubleComplex* signal, const int N) {
  int i;
  for (i = 0; i < N; ++i) {
    double theta = (double) i / (double) N * M_PI;
    signal[i].x = 1.0 * cos(10.0 * theta) + 0.5 * cos(25.0 * theta);
    signal[i].y = 1.0 * sin(10.0 * theta) + 0.5 * sin(25.0 * theta);
  }
}

static void cu_fft_double(benchmark::State& state) {
  int N = state.range(0);

  // Allocate host memory for the signal and result
  cuDoubleComplex *h_signal = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * N);
  cuDoubleComplex *h_fft = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * N);

  //  Allocate complex signal GPU device memory
  cuDoubleComplex *d_signal;
  checkCudaErrors(cudaMalloc((void ** )&d_signal, N * sizeof(cuDoubleComplex)));

  //  Init CUFFT Plan
  cufftHandle plan;
  checkCudaErrors(cufftPlan1d(&plan, N, CUFFT_Z2Z, 1));

  //  Generate signal
  generate_signal(h_signal, N);

  for (auto _ : state) {
    //  Start iteration timer
    auto start = std::chrono::high_resolution_clock::now();

    // Copy host memory to device
    checkCudaErrors(cudaMemcpy(d_signal, h_signal, N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // Transform signal to fft (inplace)
    checkCudaErrors(cufftExecZ2Z(plan, (cuDoubleComplex *)d_signal, (cuDoubleComplex *)d_signal, CUFFT_FORWARD));

    // Copy device memory to host fft
    checkCudaErrors(cudaMemcpy(h_fft, d_signal, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    //  Calculate elapsed time
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start);

    //  Set Iteration Time
    state.SetIterationTime(elapsed_seconds.count());
  }

  // Destroy CUFFT context
  checkCudaErrors(cufftDestroy(plan));

  // Cleanup memory
  checkCudaErrors(cudaFree(d_signal));
  free(h_signal);
  free(h_fft);

  //  Save statistics
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * N * sizeof(cufftComplex));
  state.SetComplexityN(N);
}
BENCHMARK(cu_fft_double)->RangeMultiplier(2)->Range(1<<10, 1<<20)->Complexity()->UseManualTime();
BENCHMARK_MAIN();

