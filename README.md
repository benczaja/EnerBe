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


