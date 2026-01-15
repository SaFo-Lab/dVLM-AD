#!/bin/bash
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH -t 20:00:00
#SBATCH -J dvlm-ad
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err


echo "Job started on $(hostname) at $(date)"
echo "Running in directory: $(pwd)"


source ~/.bashrc
source /scratch/cxiao13/miniconda3/etc/profile.d/conda.sh
conda activate dvlm

cd /weka/home/xliu316/scratchcxiao13/yingzi/LLaDA-V/train
bash scripts/llada_v_finetune.sh 1 8

echo "Job finished at $(date)"