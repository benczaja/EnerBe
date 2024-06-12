# EnerBe
Energy Benchmarks


### Build instructions

Requirements
- C and C++ Compliers (GNU preferred)
- ROCM or CUDA Compilers if compiling for AMD or NVIDIA GPUs
- CMake

Simplest example
```

git clone https://github.com/benczaja/EnerBe.git
cd EnerBe
mkdir build && cd build
cmake ../
make 
make install
```

With CUDA:
```
cmake -DENABLE_CUDA=1 ..
make && make install
```
With HIP:
```
cmake -DENABLE_HIP=1 ..
make && make install
```
With PMT:
> If you do not have PMT, PMT will be installed (https://git.astron.nl/RD/pmt)
```
cmake -DENABLE_PMT=1 ..
make && make install
```
With PMT and CUDA 
```
cmake -DENABLE_PMT=1 -DENABLE_CUDA=1 ..
```
or HIP
```
cmake -DENABLE_PMT=1 -DENABLE_HIP=1 ..
```

### Running instuctions

There will be two executables `sEnerBe` of single FP precision and `dEnerBe` for double FP precision.

Each Algorithm will be run 20 times, or until 5 minutes is reached. The Standard deviation and variation of all runs will be calculated.

List the available algorithms
```
./bin/sEnerBe -h
```
#### cblas sgemm example (with PMT enabled)

```
../bin/sEnerBe --xgemm-cblas 5000
```
output:
```
NAME: xgemm
ALGO: cblas
PRECISION: 4 bytes
OMP_THREADS: 192
MPI_RANKS: 0
NGPUs: 0
GPU ID: 99
SIZE: 5000
(RAPL) CPU_TIME: 0.241004 s
(RAPL) CPU_TIME_var: 0.00449523 s^2
(RAPL) CPU_TIME_std: 0.0670465 s
(RAPL) CPU_WATTS: 383.458 W
(RAPL) CPU_WATTS_var: 787.34 W^2
(RAPL) CPU_WATTS_std: 28.0596 W
(RAPL) CPU_JOULES: 92.6907 J
(RAPL) CPU_JOULES_var: 797.627 J^2
(RAPL) CPU_JOULES_std: 28.2423 J
NRUNS: 20
```



