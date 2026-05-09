#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J ggda
#SBATCH -o out/ggda.%J.out
#SBATCH -e err/ggda.%J.err
#SBATCH --mail-user=fernando.zhapacamacho@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=5:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint=[v100]


source /home/zhapacfp/miniforge3/etc/profile.d/conda.sh
conda activate multihopgda

wandb agent --count 1 ferzcam/multihop-gda/$1
