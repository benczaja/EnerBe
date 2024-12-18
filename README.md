# EnerBe
Energy Benchmarks


## Build instructions

Requirements
- C and C++ Compliers (GNU preferred)
- BLAS libraries (OpenBLAS preferred)
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

## Running instuctions

There will be two executables `sEnerBe` of single FP precision and `dEnerBe` for double FP precision.

Each Algorithm will be run 20 times, or until 5 minutes is reached. The Standard deviation and variation of all runs will be calculated.

List the available algorithms
```
./bin/sEnerBe -h
```
### cblas sgemm example (with PMT enabled)
> Results are from a AMD EPYC 9654 96-Core Processor (Genoa) Dual socket processor

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

## Benchmarker instructions

Requirements
- Python (pandas, seaborn)


Scripts are located here `EnerBe/benchmarker`

Simple usage:
```
python main.py -h
usage: main.py [-h] [-c CONFIG] [--concatonate [N ...]] [-s] [-p] [-r RUN [RUN ...]]

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Pass specific .json config to script
  --concatonate [N ...]
                        Concatonate multiple tmp_results.csv together
  -s, --sbatch          Create aand submit Jobscript based off info from 'bench_config.json'
  -p, --plot            Plot the Benchmark
  -r RUN [RUN ...], --run RUN [RUN ...]
                        Full command (space seperated)
```

### How to run on a node that you have allocated:

```
python /home/benjamic/EnerBe/benchmarker/main.py --run="/home/benjamic/EnerBe/bin/dEnerBe --xgemm-gpublas 10000"
```

### Run with Slurm:
You need to set up your case information in the `bench_config.json` 

Then all you need to do is:
```
python main -s
```

### Plot the results

Results will be plotted from the csvs that live in this directory: `EnerBe/benchmarker/tmp_results/results.csv`

```
python main.py -p
```
`.pngs` will be saved in the `benchmarker` directory
