#!/bin/bash

module load 2023
module load foss/2023a
module load CMake/3.26.3-GCCcore-12.3.0

cd ..

git clone https://gitlab.com/unigespc/palabos.git

cp plb/aneurysm.cpp  palabos/examples/showCases/aneurysm/
cp plb/input_1_node*  palabos/examples/showCases/aneurysm/
cp plb/CMakeLists.txt  palabos/examples/showCases/aneurysm/

cd palabos/examples/showCases/aneurysm/build

cmake ..

make -j 

cd ../

tar -xzvf aneurysm.stl.tgz
