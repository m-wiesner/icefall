#!/usr/bin/env bash
#SBATCH --job-name=mtasr-4gpu-400k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail

recipe_dir=/expscratch/mwiesner/shuffle/icefall-shuffle/egs/librispeech/MTASR
cd "$recipe_dir"
mkdir -p exp/slurm exp/shuffle_large_length_4gpu_400k
source activate.sh

export TORCH_HOME=/expscratch/mwiesner/shuffle/.cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TORCH_HOME"
start_batch=${START_BATCH:-0}
max_duration=${MAX_DURATION:-280}

python wavlm_spashl_ctc/train.py \
  --world-size 4 \
  --start-batch "$start_batch" \
  --exp-dir exp/shuffle_large_length_4gpu_400k \
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
  --total-steps 400000 \
  --freeze-iters 2000 \
  --freeze-lr 0.005 \
  --speaker-weight 1.0 \
  --original-topo false \
  --sort-strategy speech_time \
  --max-duration "$max_duration" \
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
  --num-buckets 100 \
  --num-cuts-for-bins-estimate 10000 \
  --sampler-buffer-size 20000 \
  --num-workers 5 \
  --use-fp16 true \
  --save-every-n 2000 \
  --average-period 200 \
  --tensorboard true \
  --seed 42
