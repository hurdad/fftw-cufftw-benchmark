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
./cufftwf-benchmark
./cufftw-benchmark
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
2018-12-08 17:28:43
***WARNING*** Library was built as DEBUG. Timings may be affected.
-------------------------------------------------------------------------
Benchmark                                  Time           CPU Iterations
-------------------------------------------------------------------------
cu_fft_single/1024/manual_time         19751 ns      19777 ns      37009   395.557MB/s   49.4447M items/s
cu_fft_single/2048/manual_time         23970 ns      23998 ns      29911   651.854MB/s   81.4817M items/s
cu_fft_single/4096/manual_time         27104 ns      27140 ns      26149   1.12593GB/s   144.119M items/s
cu_fft_single/8192/manual_time         32078 ns      32109 ns      21447    1.9027GB/s   243.545M items/s
cu_fft_single/16384/manual_time        53105 ns      53155 ns      12626   2.29867GB/s   294.229M items/s
cu_fft_single/32768/manual_time        87316 ns      87370 ns       7752   2.79604GB/s   357.894M items/s
cu_fft_single/65536/manual_time       160852 ns     160905 ns       4338    3.0356GB/s   388.556M items/s
cu_fft_single/131072/manual_time      311623 ns     311700 ns       2243    3.1338GB/s   401.126M items/s
cu_fft_single/262144/manual_time      548451 ns     548535 ns       1270   3.56117GB/s   455.829M items/s
cu_fft_single/524288/manual_time     1049266 ns    1049361 ns        667   3.72284GB/s   476.524M items/s
cu_fft_single/1048576/manual_time    2112286 ns    2112392 ns        333    3.6986GB/s   473.421M items/s
cu_fft_single_BigO                      2.02 N       2.02 N 
cu_fft_single_RMS                          6 %          6 % 
```
### cufft-single-unified-benchmark
```
Run on (8 X 4200 MHz CPU s)
2018-12-08 17:28:54
***WARNING*** Library was built as DEBUG. Timings may be affected.
---------------------------------------------------------------------------------
Benchmark                                          Time           CPU Iterations
---------------------------------------------------------------------------------
cu_fft_single_unified/1024/manual_time         11556 ns      11049 ns      60189   676.032MB/s    84.504M items/s
cu_fft_single_unified/2048/manual_time         14284 ns      13332 ns      51289   1093.91MB/s   136.738M items/s
cu_fft_single_unified/4096/manual_time         22913 ns      19111 ns      29399   1.33191GB/s   170.485M items/s
cu_fft_single_unified/8192/manual_time         58813 ns      41308 ns      12625   1062.69MB/s   132.836M items/s
cu_fft_single_unified/16384/manual_time       119749 ns      88636 ns       5962   1043.85MB/s   130.481M items/s
cu_fft_single_unified/32768/manual_time       640200 ns     434857 ns        936   390.503MB/s   48.8129M items/s
cu_fft_single_unified/65536/manual_time      1003725 ns     766041 ns        620   498.144MB/s    62.268M items/s
cu_fft_single_unified/131072/manual_time      591654 ns     591725 ns       1134   1.65056GB/s   211.272M items/s
cu_fft_single_unified/262144/manual_time     1007616 ns    1007723 ns        696   1.93836GB/s    248.11M items/s
cu_fft_single_unified/524288/manual_time     1768924 ns    1769052 ns        397   2.20826GB/s   282.658M items/s
cu_fft_single_unified/1048576/manual_time    3651642 ns    3651796 ns        192   2.13945GB/s   273.849M items/s
cu_fft_single_unified_BigO                      3.54 N       3.52 N 
cu_fft_single_unified_RMS                         36 %         26 % 
```
### cufft-double-benchmark
```
Run on (8 X 4200 MHz CPU s)
2018-12-08 17:29:04
***WARNING*** Library was built as DEBUG. Timings may be affected.
-------------------------------------------------------------------------
Benchmark                                  Time           CPU Iterations
-------------------------------------------------------------------------
cu_fft_double/1024/manual_time         22721 ns      22748 ns      31066    343.84MB/s     42.98M items/s
cu_fft_double/2048/manual_time         34249 ns      34276 ns      20449   456.221MB/s   57.0276M items/s
cu_fft_double/4096/manual_time         55570 ns      55603 ns      11878   562.349MB/s   70.2936M items/s
cu_fft_double/8192/manual_time         60962 ns      61000 ns      10936   1025.23MB/s   128.153M items/s
cu_fft_double/16384/manual_time        95487 ns      95523 ns       7122    1.2784GB/s   163.636M items/s
cu_fft_double/32768/manual_time       176607 ns     176644 ns       3948    1.3824GB/s   176.947M items/s
cu_fft_double/65536/manual_time       344907 ns     344974 ns       2034   1.41569GB/s   181.208M items/s
cu_fft_double/131072/manual_time      608999 ns     609062 ns       1134   1.60355GB/s   205.255M items/s
cu_fft_double/262144/manual_time     1125277 ns    1125374 ns        622   1.73568GB/s   222.167M items/s
cu_fft_double/524288/manual_time     2216979 ns    2217110 ns        317   1.76197GB/s   225.532M items/s
cu_fft_double/1048576/manual_time    4504132 ns    4504475 ns        155   1.73452GB/s   222.018M items/s
cu_fft_double_BigO                      4.29 N       4.29 N 
cu_fft_double_RMS                          4 %          4 % 
```
### cufftwf-benchmark
```
Run on (8 X 4200 MHz CPU s)
2018-12-08 17:29:15
***WARNING*** Library was built as DEBUG. Timings may be affected.
--------------------------------------------------------------------
Benchmark                             Time           CPU Iterations
--------------------------------------------------------------------
cu_fftwf/1024/manual_time         20114 ns      20147 ns      34794   388.403MB/s   48.5503M items/s
cu_fftwf/2048/manual_time         23009 ns      23044 ns      30765   679.082MB/s   84.8853M items/s
cu_fftwf/4096/manual_time         27026 ns      27062 ns      26078    1.1292GB/s   144.538M items/s
cu_fftwf/8192/manual_time         34616 ns      34652 ns      20004    1.7632GB/s    225.69M items/s
cu_fftwf/16384/manual_time        56608 ns      56662 ns      11779   2.15642GB/s   276.021M items/s
cu_fftwf/32768/manual_time        96624 ns      96675 ns       7197    2.5267GB/s   323.417M items/s
cu_fftwf/65536/manual_time       175800 ns     175857 ns       3990   2.77748GB/s   355.517M items/s
cu_fftwf/131072/manual_time      445886 ns     445955 ns       1908   2.19016GB/s   280.341M items/s
cu_fftwf/262144/manual_time      829577 ns     829653 ns        837   2.35436GB/s   301.358M items/s
cu_fftwf/524288/manual_time     1626205 ns    1626364 ns        430   2.40207GB/s   307.464M items/s
cu_fftwf/1048576/manual_time    2908991 ns    2909194 ns        242   2.68564GB/s   343.762M items/s
cu_fftwf_BigO                      2.86 N       2.86 N 
cu_fftwf_RMS                         10 %         10 % 
```
### cufftw-benchmark
```
Run on (8 X 4200 MHz CPU s)
2018-12-08 17:29:26
***WARNING*** Library was built as DEBUG. Timings may be affected.
-------------------------------------------------------------------
Benchmark                            Time           CPU Iterations
-------------------------------------------------------------------
cu_fftw/1024/manual_time         22900 ns      22936 ns      30547   682.307MB/s   42.6442M items/s
cu_fftw/2048/manual_time         35077 ns      35114 ns      19967   890.903MB/s   55.6815M items/s
cu_fftw/4096/manual_time         58327 ns      58364 ns      11334   1071.55MB/s   66.9719M items/s
cu_fftw/8192/manual_time         65203 ns      65241 ns      10220   1.87217GB/s   119.819M items/s
cu_fftw/16384/manual_time       104361 ns     104401 ns       6583   2.33939GB/s   149.721M items/s
cu_fftw/32768/manual_time       193482 ns     193526 ns       3604   2.52365GB/s   161.514M items/s
cu_fftw/65536/manual_time       481119 ns     481174 ns       1427   2.02977GB/s   129.906M items/s
cu_fftw/131072/manual_time      891661 ns     891720 ns        781   2.19043GB/s   140.188M items/s
cu_fftw/262144/manual_time     1685317 ns    1685406 ns        410   2.31781GB/s    148.34M items/s
cu_fftw/524288/manual_time     2970116 ns    2970253 ns        237   2.63037GB/s   168.344M items/s
cu_fftw/1048576/manual_time    6147474 ns    6147773 ns        113   2.54169GB/s   162.668M items/s
cu_fftw_BigO                      5.87 N       5.87 N 
cu_fftw_RMS                          6 %          6 % 
```
### fftw3f-benchmark
```
Run on (8 X 4200 MHz CPU s)
2018-12-08 17:29:37
***WARNING*** Library was built as DEBUG. Timings may be affected.
-----------------------------------------------------------------
Benchmark                          Time           CPU Iterations
-----------------------------------------------------------------
fftwf/1024/manual_time          1061 ns       1083 ns     655567   7.19347GB/s   920.764M items/s
fftwf/2048/manual_time          2534 ns       2557 ns     274154   6.02087GB/s   770.671M items/s
fftwf/4096/manual_time          7547 ns       7575 ns      92184   4.04385GB/s   517.613M items/s
fftwf/8192/manual_time         22392 ns      22422 ns      31727   2.72572GB/s   348.892M items/s
fftwf/16384/manual_time        46492 ns      46519 ns      14949   2.62562GB/s   336.079M items/s
fftwf/32768/manual_time       113423 ns     113455 ns       6185   2.15248GB/s   275.518M items/s
fftwf/65536/manual_time       250827 ns     250862 ns       2760   1.94669GB/s   249.176M items/s
fftwf/131072/manual_time      532110 ns     532157 ns       1290   1.83527GB/s   234.914M items/s
fftwf/262144/manual_time     1213591 ns    1213698 ns        572   1.60938GB/s       206M items/s
fftwf/524288/manual_time     6432471 ns    6432892 ns        107   621.845MB/s   77.7306M items/s
fftwf/1048576/manual_time   20443220 ns   20443920 ns         34   391.328MB/s    48.916M items/s
fftwf_BigO                      0.00 N^2       0.00 N^2 
fftwf_RMS                         15 %         15 % 
```
### fftw3-benchmark
```
Run on (8 X 4200 MHz CPU s)
2018-12-08 17:29:47
***WARNING*** Library was built as DEBUG. Timings may be affected.
-----------------------------------------------------------------
Benchmark                          Time           CPU Iterations
-----------------------------------------------------------------
fftw3/1024/manual_time          2253 ns       2276 ns     309133   6.77168GB/s   433.387M items/s
fftw3/2048/manual_time          5361 ns       5384 ns     127255   5.69275GB/s   364.336M items/s
fftw3/4096/manual_time         13797 ns      13821 ns      49018   4.42372GB/s   283.118M items/s
fftw3/8192/manual_time         37622 ns      37655 ns      18259   3.24462GB/s   207.656M items/s
fftw3/16384/manual_time        84457 ns      84492 ns       8428    2.8907GB/s   185.005M items/s
fftw3/32768/manual_time       186364 ns     186396 ns       3715   2.62004GB/s   167.682M items/s
fftw3/65536/manual_time       385735 ns     385771 ns       1833   2.53169GB/s   162.028M items/s
fftw3/131072/manual_time      891761 ns     891825 ns        759   2.19019GB/s   140.172M items/s
fftw3/262144/manual_time     2539636 ns    2539757 ns        267   1.53811GB/s   98.4393M items/s
fftw3/524288/manual_time    10510215 ns   10510737 ns         62   761.164MB/s   47.5728M items/s
fftw3/1048576/manual_time   38155139 ns   38156244 ns         19   419.341MB/s   26.2088M items/s
fftw3_BigO                      0.00 N^2       0.00 N^2 
fftw3_RMS                          7 %          7 % 
```
### fftw3l-benchmark
```
Run on (8 X 4200 MHz CPU s)
2018-12-08 17:31:02
***WARNING*** Library was built as DEBUG. Timings may be affected.
-----------------------------------------------------------------
Benchmark                          Time           CPU Iterations
-----------------------------------------------------------------
fftwl/1024/manual_time         26328 ns      26351 ns      26494   1.15914GB/s   37.0926M items/s
fftwl/2048/manual_time         57811 ns      57836 ns      11983   1081.11MB/s   33.7845M items/s
fftwl/4096/manual_time        129008 ns     129042 ns       5396   968.932MB/s   30.2791M items/s
fftwl/8192/manual_time        296533 ns     296567 ns       2373   843.076MB/s   26.3461M items/s
fftwl/16384/manual_time       624196 ns     624240 ns       1096    801.03MB/s   25.0322M items/s
fftwl/32768/manual_time      1362136 ns    1362196 ns        511   734.141MB/s   22.9419M items/s
fftwl/65536/manual_time      2899627 ns    2899727 ns        241   689.744MB/s   21.5545M items/s
fftwl/131072/manual_time     6767509 ns    6767711 ns         93   591.059MB/s   18.4706M items/s
fftwl/262144/manual_time    18643120 ns   18643650 ns         36   429.113MB/s   13.4098M items/s
fftwl/524288/manual_time    54239101 ns   54240414 ns         12    294.99MB/s   9.21844M items/s
fftwl/1048576/manual_time  123493768 ns  123496302 ns          5   259.122MB/s   8.09757M items/s
fftwl_BigO                      5.70 NlgN       5.70 NlgN 
fftwl_RMS                         19 %         19 % 

```
