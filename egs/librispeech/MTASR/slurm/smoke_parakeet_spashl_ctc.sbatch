#!/usr/bin/env bash
#SBATCH --job-name=mtasr-parakeet-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
recipe_dir=/expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR
cd "$recipe_dir"
mkdir -p exp/slurm
source activate.sh
export HF_HOME=/expscratch/mwiesner/shuffle/.cache/huggingface
export HF_HUB_OFFLINE=1
python parakeet_spashl_ctc/smoke_test.py
python parakeet_spashl_ctc/recipe_smoke_test.py
