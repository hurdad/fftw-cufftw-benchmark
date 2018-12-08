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
sudo reboot

sudo apt-get install libbenchmark-dev libfftw3-dev linux-tools-common linux-tools-generic
sudo cpupower frequency-set -g performance

git clone https://github.com/hurdad/fftw-cufftw-benchmark.git
cd fftw-cufftw-benchmark
make
./cufft-single-benchmark
./cufft-single-unified-benchmark
./cufft-double-benchmark
./cufftw-benchmark
./cufftwf-benchmark
./fftw3f-benchmark
./fftw3-benchmark
./fftw3l-benchmark
```

## Results
 * Intel i7-6700K @ 4.00GHz
 * NVIDIA GTX 1080
 
### cufft-single-benchmark
```
Run on (8 X 4200 MHz CPU s)
2018-12-08 17:07:03
***WARNING*** Library was built as DEBUG. Timings may be affected.
-------------------------------------------------------------------------
Benchmark                                  Time           CPU Iterations
-------------------------------------------------------------------------
cu_fft_single/1024/manual_time         21054 ns      21084 ns      33731   371.076MB/s   46.3845M items/s
cu_fft_single/2048/manual_time         22841 ns      22874 ns      30405   684.074MB/s   85.5092M items/s
cu_fft_single/4096/manual_time         25991 ns      26021 ns      26883   1.17417GB/s   150.294M items/s
cu_fft_single/8192/manual_time         32622 ns      32653 ns      21286   1.87097GB/s   239.484M items/s
cu_fft_single/16384/manual_time        52665 ns      52697 ns      12856   2.31788GB/s   296.688M items/s
cu_fft_single/32768/manual_time        88025 ns      88054 ns       7756   2.77352GB/s   355.011M items/s
cu_fft_single/65536/manual_time       160970 ns     160994 ns       4374   3.03336GB/s    388.27M items/s
cu_fft_single/131072/manual_time      309824 ns     309868 ns       2258   3.15199GB/s   403.455M items/s
cu_fft_single/262144/manual_time      547046 ns     547087 ns       1269   3.57031GB/s       457M items/s
cu_fft_single/524288/manual_time     1050074 ns    1050086 ns        667   3.71998GB/s   476.157M items/s
cu_fft_single/1048576/manual_time    2091058 ns    2091007 ns        334   3.73615GB/s   478.227M items/s
cu_fft_single_BigO                      2.01 N       2.01 N 
cu_fft_single_RMS                          6 %          6 % 
```
### cufft-single-unified-benchmark
```
Run on (8 X 4200 MHz CPU s)
2018-12-08 17:07:14
***WARNING*** Library was built as DEBUG. Timings may be affected.
---------------------------------------------------------------------------------
Benchmark                                          Time           CPU Iterations
---------------------------------------------------------------------------------
cu_fft_single_unified/1024/manual_time         11546 ns      11098 ns      59830   676.613MB/s   84.5767M items/s
cu_fft_single_unified/2048/manual_time         14409 ns      13442 ns      50747   1084.39MB/s   135.549M items/s
cu_fft_single_unified/4096/manual_time         23108 ns      19091 ns      29598   1.32068GB/s   169.047M items/s
cu_fft_single_unified/8192/manual_time         56940 ns      39837 ns      12756   1097.65MB/s   137.206M items/s
cu_fft_single_unified/16384/manual_time       147362 ns      99601 ns       4871   848.254MB/s   106.032M items/s
cu_fft_single_unified/32768/manual_time       444252 ns     307110 ns       2448   562.744MB/s    70.343M items/s
cu_fft_single_unified/65536/manual_time      1053706 ns     786973 ns        703   474.516MB/s   59.3144M items/s
cu_fft_single_unified/131072/manual_time      595064 ns     595116 ns       1094   1.64111GB/s   210.062M items/s
cu_fft_single_unified/262144/manual_time     1008534 ns    1008567 ns        698    1.9366GB/s   247.885M items/s
cu_fft_single_unified/524288/manual_time     1765527 ns    1765528 ns        396   2.21251GB/s   283.202M items/s
cu_fft_single_unified/1048576/manual_time    3811515 ns    3811488 ns        181   2.04971GB/s   262.363M items/s
cu_fft_single_unified_BigO                      3.65 N       3.63 N 
cu_fft_single_unified_RMS                         34 %         24 % 
```
### cufft-double-benchmark
```
Run on (8 X 4200 MHz CPU s)
2018-12-08 17:07:26
***WARNING*** Library was built as DEBUG. Timings may be affected.
--------------------------------------------------------------------------
Benchmark                                   Time           CPU Iterations
--------------------------------------------------------------------------
cu_fft_double/1024/manual_time          35065 ns      35096 ns      29323   222.801MB/s   27.8502M items/s
cu_fft_double/2048/manual_time          35684 ns      35716 ns      20576   437.868MB/s   54.7336M items/s
cu_fft_double/4096/manual_time          56380 ns      56412 ns      12039   554.277MB/s   69.2846M items/s
cu_fft_double/8192/manual_time          64174 ns      64207 ns      10929   973.917MB/s    121.74M items/s
cu_fft_double/16384/manual_time         98310 ns      98344 ns       6988   1.24168GB/s   158.936M items/s
cu_fft_double/32768/manual_time        187666 ns     187702 ns       3795   1.30093GB/s   166.519M items/s
cu_fft_double/65536/manual_time        362150 ns     362185 ns       1992   1.34828GB/s    172.58M items/s
cu_fft_double/131072/manual_time       613921 ns     613941 ns       1065    1.5907GB/s   203.609M items/s
cu_fft_double/262144/manual_time      1145909 ns    1145915 ns        618   1.70443GB/s   218.167M items/s
cu_fft_double/524288/manual_time      2271075 ns    2271106 ns        310      1.72GB/s    220.16M items/s
cu_fft_double/1048576/manual_time     4672662 ns    4672796 ns        152   1.67196GB/s   214.011M items/s
cu_fft_double/2097152/manual_time     9257956 ns    9257965 ns         69   1.68774GB/s    216.03M items/s
cu_fft_double/4194304/manual_time    18930000 ns   18929185 ns         36   1.65082GB/s   211.305M items/s
cu_fft_double/8388608/manual_time    39096755 ns   39091028 ns         17    1.5986GB/s   204.621M items/s
cu_fft_double/16777216/manual_time   85581675 ns   85569330 ns          8   1.46059GB/s   186.956M items/s
cu_fft_double/33554432/manual_time  189176253 ns  189165474 ns          3   1.32152GB/s   169.154M items/s
cu_fft_double/67108864/manual_time  547746583 ns  547589683 ns          1   934.739MB/s   116.842M items/s
cu_fft_double_BigO                       0.29 NlgN       0.29 NlgN 
cu_fft_double_RMS                          35 %         35 % 
```
### cufftw-benchmark
```
Run on (8 X 4200 MHz CPU s)
2018-12-08 17:07:56
***WARNING*** Library was built as DEBUG. Timings may be affected.
--------------------------------------------------------------------
Benchmark                             Time           CPU Iterations
--------------------------------------------------------------------
cu_fftw/1024/manual_time          23235 ns      23264 ns      30001   672.491MB/s   42.0307M items/s
cu_fftw/2048/manual_time          35650 ns      35684 ns      19940   876.568MB/s   54.7855M items/s
cu_fftw/4096/manual_time          58700 ns      58732 ns      11619   1064.74MB/s   66.5466M items/s
cu_fftw/8192/manual_time          65454 ns      65487 ns      10224   1.86499GB/s   119.359M items/s
cu_fftw/16384/manual_time        104699 ns     104733 ns       6565   2.33182GB/s   149.237M items/s
cu_fftw/32768/manual_time        202718 ns     202751 ns       3545   2.40867GB/s   154.155M items/s
cu_fftw/65536/manual_time        401516 ns     401546 ns       1805   2.43219GB/s    155.66M items/s
cu_fftw/131072/manual_time       721845 ns     721860 ns       1012   2.70574GB/s   173.167M items/s
cu_fftw/262144/manual_time      1445658 ns    1445672 ns        493   2.70206GB/s   172.932M items/s
cu_fftw/524288/manual_time      3027875 ns    3028005 ns        238   2.58019GB/s   165.132M items/s
cu_fftw/1048576/manual_time     6068151 ns    6068300 ns        108   2.57492GB/s   164.795M items/s
cu_fftw/2097152/manual_time    12481845 ns   12481516 ns         51   2.50364GB/s   160.233M items/s
cu_fftw/4194304/manual_time    26510072 ns   26509592 ns         26   2.35759GB/s   150.886M items/s
cu_fftw/8388608/manual_time    56393531 ns   56391875 ns         11   2.21657GB/s    141.86M items/s
cu_fftw/16777216/manual_time  125259734 ns  125257136 ns          5   1.99585GB/s   127.735M items/s
cu_fftw/33554432/manual_time  316405805 ns  316388813 ns          2   1.58025GB/s   101.136M items/s
cu_fftw/67108864/manual_time  877621386 ns  877526933 ns          1   1.13944GB/s   72.9244M items/s
cu_fftw_BigO                       0.47 NlgN       0.47 NlgN 
cu_fftw_RMS                          35 %         35 % 
```
### cufftwf-benchmark
```
Run on (8 X 4200 MHz CPU s)
2018-12-08 17:08:26
***WARNING*** Library was built as DEBUG. Timings may be affected.
--------------------------------------------------------------------
Benchmark                             Time           CPU Iterations
--------------------------------------------------------------------
cu_fftwf/1024/manual_time         18332 ns      18364 ns      38625   426.162MB/s   53.2703M items/s
cu_fftwf/2048/manual_time         23294 ns      23324 ns      29757   670.777MB/s   83.8471M items/s
cu_fftwf/4096/manual_time         27340 ns      27370 ns      25975   1.11622GB/s   142.877M items/s
cu_fftwf/8192/manual_time         34915 ns      34944 ns      20265   1.74812GB/s    223.76M items/s
cu_fftwf/16384/manual_time        58961 ns      58994 ns      11508   2.07035GB/s   265.005M items/s
cu_fftwf/32768/manual_time        96746 ns      96779 ns       7205   2.52351GB/s    323.01M items/s
cu_fftwf/65536/manual_time       176556 ns     176587 ns       3881   2.76559GB/s   353.995M items/s
cu_fftwf/131072/manual_time      347054 ns     347083 ns       2042   2.81387GB/s   360.175M items/s
cu_fftwf/262144/manual_time      621626 ns     621647 ns       1054   3.14196GB/s   402.171M items/s
cu_fftwf/524288/manual_time     1312393 ns    1312536 ns        509   2.97643GB/s   380.983M items/s
cu_fftwf/1048576/manual_time    2883211 ns    2883170 ns        239   2.70965GB/s   346.836M items/s
cu_fftwf_BigO                      0.14 NlgN       0.14 NlgN 
cu_fftwf_RMS                          6 %          6 % 
```
### fftw3f-benchmark
```
Run on (8 X 4200 MHz CPU s)
2018-12-08 17:09:06
***WARNING*** Library was built as DEBUG. Timings may be affected.
-----------------------------------------------------------------
Benchmark                          Time           CPU Iterations
-----------------------------------------------------------------
fftwf/1024/manual_time          1090 ns       1114 ns     645024   6.99656GB/s   895.559M items/s
fftwf/2048/manual_time          2569 ns       2592 ns     266567    5.9393GB/s    760.23M items/s
fftwf/4096/manual_time          7433 ns       7456 ns      92567   4.10592GB/s   525.557M items/s
fftwf/8192/manual_time         23663 ns      23697 ns      30974   2.57934GB/s   330.156M items/s
fftwf/16384/manual_time        49414 ns      49441 ns      14822   2.47034GB/s   316.204M items/s
fftwf/32768/manual_time       113503 ns     113529 ns       6379   2.15097GB/s   275.324M items/s
fftwf/65536/manual_time       250024 ns     250044 ns       2790   1.95294GB/s   249.976M items/s
fftwf/131072/manual_time      562284 ns     562321 ns       1274   1.73678GB/s   222.307M items/s
fftwf/262144/manual_time     1258312 ns    1258376 ns        541   1.55218GB/s   198.679M items/s
fftwf/524288/manual_time     7201441 ns    7201379 ns        101   555.444MB/s   69.4305M items/s
fftwf/1048576/manual_time   22090857 ns   22090558 ns         32   362.141MB/s   45.2676M items/s
fftwf_BigO                      0.00 N^2       0.00 N^2 
fftwf_RMS                         17 %         17 % 
```
### fftw3-benchmark
```
Run on (8 X 4200 MHz CPU s)
2018-12-08 17:08:36
***WARNING*** Library was built as DEBUG. Timings may be affected.
------------------------------------------------------
Benchmark               Time           CPU Iterations
------------------------------------------------------
fftw3/1024           2321 ns       2321 ns     309309   6.57321GB/s   420.685M items/s
fftw3/2048           5310 ns       5310 ns     129278   5.74768GB/s   367.851M items/s
fftw3/4096          14043 ns      14042 ns      49386    4.3465GB/s   278.176M items/s
fftw3/8192          36087 ns      36086 ns      18289   3.38278GB/s   216.498M items/s
fftw3/16384         84054 ns      84052 ns       8058   2.90465GB/s   185.898M items/s
fftw3/32768        188017 ns     188007 ns       3650   2.59715GB/s   166.217M items/s
fftw3/65536        395529 ns     395519 ns       1741   2.46907GB/s    158.02M items/s
fftw3/131072       903651 ns     903622 ns        785   2.16144GB/s   138.332M items/s
fftw3/262144      2605404 ns    2605318 ns        286   1.49934GB/s   95.9576M items/s
fftw3/524288     11992469 ns   11991997 ns         52   667.112MB/s   41.6945M items/s
fftw3/1048576    34002167 ns   34001273 ns         19   470.571MB/s   29.4107M items/s
fftw3/2097152    77218257 ns   77216354 ns          8    414.42MB/s   25.9012M items/s
fftw3/4194304   196512141 ns  196506941 ns          3   325.688MB/s   20.3555M items/s
fftw3/8388608   505880113 ns  505866993 ns          1   253.031MB/s   15.8144M items/s
fftw3/16777216 1060022395 ns 1059986269 ns          1   241.513MB/s   15.0945M items/s
fftw3/33554432 2285941151 ns 2285870087 ns          1   223.985MB/s    13.999M items/s
fftw3/67108864 6287768087 ns 6287545284 ns          1   162.862MB/s   10.1789M items/s
fftw3_BigO           3.39 NlgN       3.39 NlgN 
fftw3_RMS              30 %         30 % 
```
### fftw3l-benchmark
```
Run on (8 X 4200 MHz CPU s)
2018-12-08 17:14:00
***WARNING*** Library was built as DEBUG. Timings may be affected.
-----------------------------------------------------------------
Benchmark                          Time           CPU Iterations
-----------------------------------------------------------------
fftwf/1024/manual_time          1080 ns       1103 ns     616574   7.06464GB/s   904.274M items/s
fftwf/2048/manual_time          2629 ns       2652 ns     266945   5.80462GB/s   742.991M items/s
fftwf/4096/manual_time          7695 ns       7719 ns      90351   3.96581GB/s   507.624M items/s
fftwf/8192/manual_time         23184 ns      23211 ns      31559   2.63264GB/s   336.978M items/s
fftwf/16384/manual_time        47709 ns      47742 ns      14656   2.55864GB/s   327.506M items/s
fftwf/32768/manual_time       111522 ns     111557 ns       5873   2.18916GB/s   280.213M items/s
fftwf/65536/manual_time       278366 ns     278400 ns       2542    1.7541GB/s   224.524M items/s
fftwf/131072/manual_time      602951 ns     602985 ns       1080   1.61964GB/s   207.314M items/s
fftwf/262144/manual_time     1316571 ns    1316605 ns        534   1.48349GB/s   189.887M items/s
fftwf/524288/manual_time     6686395 ns    6686331 ns        105    598.23MB/s   74.7787M items/s
fftwf/1048576/manual_time   20342179 ns   20341920 ns         41   393.272MB/s   49.1589M items/s
fftwf_BigO                      0.00 N^2       0.00 N^2 
fftwf_RMS                         18 %         18 % 
```
