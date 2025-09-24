#!/bin/bash
#-------------Script SBATCH - NLHPC ----------------
#SBATCH -J training_MPhyVAENN
#SBATCH -p main
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mail-user=sjurado@das.uchile.cl
#SBATCH --mail-type=ALL
#SBATCH -o training_MPhyVAENN_%j.err.out
#SBATCH -e training_MPhyVAENN_%j.err.out

#-----------------Toolchain---------------------------
# ----------------Modulos----------------------------
# Activate CONDA environments
eval "$(conda shell.bash hook)"
conda activate rocm-env

# ----------------Comando--------------------------
python Training_MPhyVAE.py
