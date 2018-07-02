CC      = g++
CFLAGS  = -std=c++11 -O0 -g -Wall -Wextra -Wshadow -pedantic -I/usr/local/cuda-9.2/targets/x86_64-linux/include/
LDFLAGS = -L=/usr/local/cuda-9.2/targets/x86_64-linux/lib/ -lbenchmark -lm

all: fftw3-benchmark cufftw-benchmark

fftw3-benchmark: fftw3-benchmark.o
	$(CC) -o $@ $^ $(LDFLAGS) -lfftw3

fftw3-benchmark.o: fftw3-benchmark.cc
	$(CC) -c $(CFLAGS) $<
	
cufftw-benchmark: cufftw-benchmark.o
	$(CC) -o $@ $^ $(LDFLAGS) -lcufftw

cufftw-benchmark.o: cufftw-benchmark.cc
	$(CC) -c $(CFLAGS) $<

.PHONY: clean

clean:
	rm *.o fftw3-benchmark cufftw-benchmark