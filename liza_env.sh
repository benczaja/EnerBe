#!/bin/bash

source /cvmfs/software.eessi.io/versions/2023.06/init/bash 
module load foss/2023b

unset LIBRARY_PATH
unset LD_LIBRARY_PATH

if [ $1 == "intel" ]
then
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64
    # this probably needs to be fixed on LIZA
    export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
    # BLAS
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/intel/skylake_avx512/software/FlexiBLAS/3.3.1-GCC-13.2.0/lib
    # LIBSTD
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/intel/skylake_avx512/software/GCCcore/13.2.0/lib64
fi
if [ $1 == "amd" ]
then
    # this probably needs to be fixed on LIZA
    export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
    # BLAS
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/intel/skylake_avx512/software/FlexiBLAS/3.3.1-GCC-13.2.0/lib
    # LIBSTD
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/intel/skylake_avx512/software/GCCcore/13.2.0/lib64
fi
if [ $1 == "arm" ]
then
    # this probably needs to be fixed on LIZA
    export LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LIBRARY_PATH
    # CUDA
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64
    # LIBSTD
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/software.eessi.io/versions/2023.06/software/linux/aarch64/neoverse_v1/software/GCCcore/13.2.0/lib64
    # OpenBLAS
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/software.eessi.io/versions/2023.06/software/linux/aarch64/neoverse_v1/software/FlexiBLAS/3.3.1-GCC-13.2.0/lib
    #OpenMPI
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/software.eessi.io/versions/2023.06/software/linux/aarch64/neoverse_v1/software/OpenMPI/4.1.6-GCC-13.2.0/lib
fi

