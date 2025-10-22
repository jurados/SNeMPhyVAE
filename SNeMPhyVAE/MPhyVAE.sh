#!/bin/bash
#-------------Script SBATCH - NLHPC ----------------
#SBATCH -J training_MPhyVAENN
#SBATCH -p mi210
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH -c 4
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-user=sjurado@das.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -o logs/training_%j.err.out
#SBATCH -e logs/training_%j.err.out

#-----------------Toolchain---------------------------
# ----------------Modulos----------------------------
# Activate CONDA environments
eval "$(conda shell.bash hook)"
conda activate rocm-env

# ---------------Variables---------------------------
LatentSpace_Value=${1:-3}

# ----------------Comando--------------------------
python model/training.py -ls $LatentSpace_Value
