#!/usr/bin/env bash
#SBATCH --job-name=mtasr-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail

recipe_dir=/expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR
cd "$recipe_dir"
smoke_exp_dir="exp/smoke_shuffle_large_length_unfreeze_${SLURM_JOB_ID}"
mkdir -p exp/slurm "$smoke_exp_dir"
source activate.sh

export TORCH_HOME=/expscratch/mwiesner/shuffle/.cache/torch
mkdir -p "$TORCH_HOME"

python wavlm_spashl_ctc/train.py \
  --world-size 1 \
  --num-epochs 1 \
  --max-train-batches 2 \
  --max-valid-batches 1 \
  --exp-dir "$smoke_exp_dir" \
  --bpe-model data/lang_bpe_5000/bpe.model \
  --lang-dir data/lang_bpe_5000 \
  --use-large true \
  --factorized-token-speaker-output true \
  --downsample false \
  --use-layer-norm false \
  --max-num-spks 4 \
  --max-overlap 3 \
  --collar 64000 \
  --interval-model fixed \
  --beam-size 30 \
  --total-max-overlaps 6000 \
  --base-lr 0.00005 \
  --pct-start 0.15 \
  --total-steps 2 \
  --freeze-iters 1 \
  --freeze-lr 0.005 \
  --speaker-weight 1.0 \
  --original-topo false \
  --sort-strategy speech_time \
  --max-duration 30 \
  --max-splices 10 \
  --max-splice-duration 50 \
  --max-unique 4 \
  --overlap 1.0 \
  --min-overlap 0.5 \
  --drift 0.5 \
  --reverb true \
  --normalize-loudness true \
  --max-snr 30 \
  --allow-self-overlap false \
  --duration-increment 0.00001 \
  --num-buckets 10 \
  --num-cuts-for-bins-estimate 100 \
  --sampler-buffer-size 200 \
  --num-workers 1 \
  --use-fp16 true \
  --tensorboard false \
  --seed 42
