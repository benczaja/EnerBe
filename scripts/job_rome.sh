#!/bin/bash

#SBATCH -p rome
#SBATCH -t 00:59:00
#SBATCH --ntasks=128
#SBATCH --exclusive 

module load 2022 foss/2022a
module load SciPy-bundle/2022.05-foss-2022a
module load pmt/1.2.0-GCCcore-11.3.0  

cd /home/benjamic/EnerBe/scripts/
./benchmarker.py