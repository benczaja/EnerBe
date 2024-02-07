# EnerBe
Energy Benchmarks



### Build insttructions

Simplist example
```
module load 2023 foss/2023a
module load CMake/3.26.3-GCCcore-12.3.0 
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
```
cmake -DENABLE_PMT=1 ..
make && make install
```

### If you need to build PMT
go to the root EnerBe directory
```
git clone --recursive https://git.astron.nl/RD/pmt.git
cd pmt && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../ ..
make install
```
Then set the `-DPMT_DIR` with CMake
example
```
cmake -DENABLE_PMT=1 -DPMT_DIR=/home/benjamic/EnerBe/pmt ../
```

