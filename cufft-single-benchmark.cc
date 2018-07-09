#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>
#include <benchmark/benchmark.h>
#include <math.h>

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

static void cu_fftw(benchmark::State& state) {
	int N = state.range(0);
	for (auto _ : state) {

		// Allocate host memory for the signal
		cufftComplex *h_signal = (cufftComplex *)malloc(sizeof(cufftComplex) * N);
		cufftComplex *h_fft = (cufftComplex *)malloc(sizeof(cufftComplex) * N);

		// Initalize the memory for the signal
		state.PauseTiming();
		generate_signal(h_signal, N);
		state.ResumeTiming();

		//  Allocate complex signal GPU device memory
		cufftComplex *d_signal;
		checkCudaErrors(cudaMalloc((void **)&d_signal, N*sizeof(cufftComplex)));
		
		// Copy host memory to device
		checkCudaErrors(cudaMemcpy(d_signal, h_signal, N*sizeof(cufftComplex), cudaMemcpyHostToDevice));
		cufftHandle plan;
		checkCudaErrors(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

		// Transform signal to fft (inplace)
		checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD));

		// Copy device memory to host fft
		checkCudaErrors(cudaMemcpy(h_fft, d_signal, N*sizeof(cufftComplex), cudaMemcpyDeviceToHost));
			
		// Destroy CUFFT context
		checkCudaErrors(cufftDestroy(plan));
				   
		// cleanup memory
		checkCudaErrors(cudaFree(d_signal));
		free(h_signal);
		free(h_fft);
	}
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
	state.SetBytesProcessed(
			static_cast<int64_t>(state.iterations()) * N
					* sizeof(cufftComplex));
	state.SetComplexityN(N);
}
BENCHMARK(cu_fftw)->RangeMultiplier(2)->Range(1<<10, 1<<26)->Complexity();
BENCHMARK_MAIN();



