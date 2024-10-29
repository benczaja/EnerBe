#!/bin/bash

#Snellius stuff
#module load 2023
module load foss/2023b
#module load CMake/3.26.3-GCCcore-12.3.0
#module load pmt/1.2.0-GCCcore-12.3.0

# in case you are not working with modules
export LIBRARY_PATH=$LIBRARY_PATH:/home/benjamic/EnerBe/pmt/lib
export LIBRARY_PATH=$LIBRARY_PATH:/opt/rocm/lib
export LIBRARY_PATH=$LIBRARY_PATH:/cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/amd/zen3/software/GCCcore/13.2.0/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/amd/zen3/software/OpenMPI/4.1.6-GCC-13.2.0/lib


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/benjamic/EnerBe/pmt/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/amd/zen3/software/GCCcore/13.2.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/amd/zen3/software/OpenMPI/4.1.6-GCC-13.2.0/lib


cd ..

git clone https://gitlab.com/unigespc/palabos.git

cp plb/aneurysm.cpp  palabos/examples/showCases/aneurysm/
cp plb/CMakeLists.txt  palabos/examples/showCases/aneurysm/
cp plb/input_1_node*  ../bin

cd palabos/examples/showCases/aneurysm/build

cmake -DENABLE_PMT=1 -DPMT_ROOT=/home/benjamic/EnerBe/pmt -DCMAKE_PREFIX_PATH=/home/benjamic/EnerBe/pmt ..

make -j 

cd ../

tar -xzvf aneurysm.stl.tgz

cp aneurysm ../../../../../bin/aneurysm
cp aneurysm.stl ../../../../../bin/

cd ../../../../../
