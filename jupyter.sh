#!/bin/bash

#SBATCH --job-name=jupyter_%j
#SBATCH --output=jupyter_%j.out
#SBATCH --partition=vcpu,hpcpu # -p
#SBATCH --cpus-per-task=1 # -c
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=08:00:00 
 
# Initialize conda:
eval "$(conda shell.bash hook)"
 
if [ "$#" -eq 0 ]; then
	port=61112
else
	port=$1
fi

# some command to activate a specific conda environment or whatever:
# conda activate base

jupyter-lab --port=$port

