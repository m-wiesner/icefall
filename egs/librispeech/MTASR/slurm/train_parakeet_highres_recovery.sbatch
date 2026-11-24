#!/usr/bin/env bash
#SBATCH --job-name=mtasr-parakeet-highres
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail
recipe_dir=${RECIPE_DIR:-$SLURM_SUBMIT_DIR}
export HIGH_RESOLUTION_FUSION=true
export RECOVERY_ADAPTER_BLOCKS=${RECOVERY_ADAPTER_BLOCKS:-2}
export WORLD_SIZE=${WORLD_SIZE:-2}
export MAX_DURATION=${MAX_DURATION:-150}
export EXP_DIR=${EXP_DIR:-exp/parakeet_spashl_highres_recovery_50hz}
exec bash "$recipe_dir/slurm/train_parakeet_shuffle_large_length.sbatch"
