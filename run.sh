#!/bin/bash
# 
#SBATCH --job-name=initial
##SBATCH --output
#SBATCH --partition=mbzuai
#SBATCH --time=03:00:00
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gpus=1
#

/nfs/users/ext_shikhar.srivastava/miniconda3/envs/avalanche-dev-env/bin/python -u attribute_vectors.py
