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
#### cblas dgemm example

```
./bin/sEnerBe --xgemm-cblas  2000
```
output:
```
NAME: xgemm
ALGO: cblas
PRECISION: 4 bytes
OMP_THREADS: 24
MPI_RANKS: 0
NGPUs: 0
GPU ID: 99
SIZE: 2000
(RAPL) CPU_TIME: 0.100102 s
(RAPL) CPU_TIME_var: 1.95394e-11 s^2
(RAPL) CPU_TIME_std: 4.42033e-06 s
(RAPL) CPU_WATTS: 0 W
(RAPL) CPU_WATTS_var: 0 W^2
(RAPL) CPU_WATTS_std: 0 W
(RAPL) CPU_JOULES: 0 J
(RAPL) CPU_JOULES_var: 0 J^2
(RAPL) CPU_JOULES_std: 0 J
NRUNS: 20
```



