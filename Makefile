CC      = g++
CFLAGS  = -std=c++11 -O0 -g -Wall -Wextra -Wshadow -pedantic -I/usr/local/cuda-9.2/targets/x86_64-linux/include/
LDFLAGS = -L=/usr/local/cuda-9.2/targets/x86_64-linux/lib/

all: fftw-benchmark cufftw-benchmark

fftw-benchmark: fftw-benchmark.o
	$(CC) -o $@ $^ $(LDFLAGS) -lfftw3

fftw-benchmark.o: fftw-benchmark.cc
	$(CC) -c $(CFLAGS) $<
	
cufftw-benchmark: cufftw-benchmark.o
	$(CC) -o $@ $^ $(LDFLAGS) -lcufftw

cufftw-benchmark.o: cufftw-benchmark.cc
	$(CC) -c $(CFLAGS) $<

.PHONY: clean

clean:
	rm *.o fftw-benchmark cufftw-benchmark