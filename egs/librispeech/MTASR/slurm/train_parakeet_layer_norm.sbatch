#!/usr/bin/env bash
#SBATCH --job-name=mtasr-parakeet-ln
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail

export PARAKEET_CONVOLUTION_NORM=layer_norm
export SPEAKER_LAYER=${SPEAKER_LAYER:-18}
export EXP_DIR=${EXP_DIR:-exp/parakeet_spashl_learned_phase_layer_norm}

exec bash /expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR/slurm/train_parakeet_shuffle_large_length.sbatch
