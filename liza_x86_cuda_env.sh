#!/bin/bash

source /cvmfs/software.eessi.io/versions/2023.06/init/bash 


export LIBRARY_PATH=$LIBRARY_PATH:/home/benjamic/EnerBe/pmt/lib
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/amd/zen3/software/GCCcore/13.2.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/amd/zen3/software/OpenMPI/4.1.6-GCC-13.2.0/lib

#export LD_LIBRARY_PATH=$LIBRARY_PATH