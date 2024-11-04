#!/bin/bash


source /cvmfs/software.eessi.io/versions/2023.06/init/bash 

export LIBRARY_PATH=$LIBRARY_PATH:/home/benjamic/EnerBe/pmt/lib
export LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LIBRARY_PATH

export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/benjamic/EnerBe/pmt/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/software.eessi.io/versions/2023.06/software/linux/aarch64/neoverse_v1/software/GCCcore/13.2.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/software.eessi.io/versions/2023.06/software/linux/aarch64/neoverse_v1/software/FlexiBLAS/3.3.1-GCC-13.2.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/software.eessi.io/versions/2023.06/software/linux/aarch64/neoverse_v1/software/OpenMPI/4.1.6-GCC-13.2.0/lib
