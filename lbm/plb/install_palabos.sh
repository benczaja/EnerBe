#!/bin/bash

#Snellius stuff
module load 2023
module load foss/2023a
module load CMake/3.26.3-GCCcore-12.3.0
module load pmt/1.2.0-GCCcore-12.3.0
#LIZA stuff
#export LIBRARY_PATH=$LIBRARY_PATH:/home/benjamic/EnerBe/pmt/lib
#cmake -DENABLE_PMT=1 -DPMT_ROOT=/home/benjamic/EnerBe/pmt -DCMAKE_PREFIX_PATH=/cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/generic/software/OpenMPI/4.1.4-GCC-12.2.0 ..
#Needed at runtime (maybe could consider setting RPATH)
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/benjamic/EnerBe/pmt/lib

cd ..

git clone https://gitlab.com/unigespc/palabos.git

cp plb/aneurysm.cpp  palabos/examples/showCases/aneurysm/
cp plb/CMakeLists.txt  palabos/examples/showCases/aneurysm/
cp plb/input_1_node*  ../bin

cd palabos/examples/showCases/aneurysm/build

cmake -DENABLE_PMT=1 ..
# LIZA
#cmake -DENABLE_PMT=1 -DPMT_ROOT=/home/benjamic/EnerBe/pmt -DCMAKE_PREFIX_PATH=/cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/generic/software/OpenMPI/4.1.4-GCC-12.2.0 ..

make -j 

cd ../

tar -xzvf aneurysm.stl.tgz

cp aneurysm ../../../../../bin/aneurysm_pmt
cp aneurysm.stl ../../../../../bin/

cd ../../../../../