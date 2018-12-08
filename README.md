# fftw-cufftw-benchmark

Benchmark for popular fft libaries - fftw | cufftw | cufft

## Dependancies
 * fftw [http://www.fftw.org/]
 * cuda + cufft/cufftw [https://developer.nvidia.com/cuda-downloads]
 * benchmark [https://github.com/google/benchmark]

## Ubuntu Quickstart 18.04
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
sudo apt-get install libbenchmark-dev libfftw3-dev

git clone https://github.com/hurdad/fftw-cufftw-benchmark.git
cd fftw-cufftw-benchmark
make
./cufft-single-benchmark
./cufft-single-unified-benchmark
./cufft-double-benchmark
./cufftw-benchmark
./cufftwf-benchmark
./fftw3-benchmark
./fftw3f-benchmark
./fftw3l-benchmark
```
