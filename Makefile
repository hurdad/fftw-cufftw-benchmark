CC      = g++
CFLAGS  = -std=c++11 -O0 -g -Wall -Wextra -Wshadow -pedantic -I/usr/local/cuda-9.2/targets/x86_64-linux/include/ -I/usr/local/cuda-9.2/samples/common/inc

LDFLAGS = -L=/usr/local/cuda-9.2/targets/x86_64-linux/lib/ -lbenchmark -lm

all: fftw3-benchmark fftw3f-benchmark fftw3l-benchmark cufftw-benchmark cufftwf-benchmark cufft-single-benchmark cufft-double-benchmark

fftw3-benchmark: fftw3-benchmark.o
	$(CC) -o $@ $^ $(LDFLAGS) -lfftw3

fftw3-benchmark.o: fftw3-benchmark.cc
	$(CC) -c $(CFLAGS) $<
	
fftw3f-benchmark: fftw3f-benchmark.o
	$(CC) -o $@ $^ $(LDFLAGS) -lfftw3f

fftw3f-benchmark.o: fftw3f-benchmark.cc
	$(CC) -c $(CFLAGS) $<
	
fftw3l-benchmark: fftw3l-benchmark.o
	$(CC) -o $@ $^ $(LDFLAGS) -lfftw3l

fftw3l-benchmark.o: fftw3l-benchmark.cc
	$(CC) -c $(CFLAGS) $<
	
cufftw-benchmark: cufftw-benchmark.o
	$(CC) -o $@ $^ $(LDFLAGS) -lcufftw

cufftw-benchmark.o: cufftw-benchmark.cc
	$(CC) -c $(CFLAGS) $<
	
cufftwf-benchmark: cufftwf-benchmark.o
	$(CC) -o $@ $^ $(LDFLAGS) -lcufftw

cufftwf-benchmark.o: cufftwf-benchmark.cc
	$(CC) -c $(CFLAGS) $<
	
cufft-single-benchmark: cufft-single-benchmark.o
	$(CC) -o $@ $^ $(LDFLAGS) -lcufft -lcudart

cufft-single-benchmark.o: cufft-single-benchmark.cc
	$(CC) -c $(CFLAGS) $<
	
cufft-double-benchmark: cufft-double-benchmark.o
	$(CC) -o $@ $^ $(LDFLAGS) -lcufft -lcudart

cufft-double-benchmark.o: cufft-double-benchmark.cc
	$(CC) -c $(CFLAGS) $<

.PHONY: clean

clean:
	rm *.o fftw3-benchmark fftw3f-benchmark fftw3l-benchmark cufftw-benchmark cufftwf-benchmark cufft-single-benchmark cufft-double-benchmark