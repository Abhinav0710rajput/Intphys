#!/bin/bash
#SBATCH -A torch_pr_36_mren
#SBATCH -p l40s_mren
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ar10026@nyu.edu

set -eo pipefail

echo "Job started on $(hostname)"
echo "CUDA devices:"
nvidia-smi


source ~/.bashrc

conda activate torchenv

python video_edit.py

echo "Job finished"
