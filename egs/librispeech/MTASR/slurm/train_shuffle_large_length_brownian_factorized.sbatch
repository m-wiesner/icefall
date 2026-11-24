#!/usr/bin/env bash
#SBATCH --job-name=mtasr-wavlm-brown-factor
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail

recipe_dir=${RECIPE_DIR:-$SLURM_SUBMIT_DIR}
cd "$recipe_dir"
mkdir -p exp/slurm
source activate.sh

export TORCH_HOME=${TORCH_HOME:-/expscratch/mwiesner/shuffle/.cache/torch}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$recipe_dir/.venv/lib/python3.11/site-packages/torch/lib:$recipe_dir/.venv/lib/python3.11/site-packages/k2/lib64:/expscratch/mwiesner/shuffle/.tools/cuda118/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

start_batch=${START_BATCH:-0}
max_duration=${MAX_DURATION:-280}
max_train_batches=${MAX_TRAIN_BATCHES:-0}
exp_dir=${EXP_DIR:-exp/shuffle_large_length_brownian_factorized}
mkdir -p "$exp_dir" "$TORCH_HOME"

python wavlm_spashl_ctc/train.py \
  --world-size 1 \
  --start-batch "$start_batch" \
  --exp-dir "$exp_dir" \
  --bpe-model data/lang_bpe_5000/bpe.model \
  --lang-dir data/lang_bpe_5000 \
  --use-large true \
  --factorized-token-speaker-output true \
  --factorized-ctc-loss true \
  --downsample false \
  --use-layer-norm false \
  --max-num-spks 4 \
  --max-overlap 3 \
  --collar 64000 \
  --edge-collar 0 \
  --interval-model brownian_bridge \
  --beam-size 30 \
  --total-max-overlaps 6000 \
  --base-lr 0.00005 \
  --pct-start 0.15 \
  --total-steps 200000 \
  --freeze-iters 2000 \
  --freeze-lr 0.005 \
  --speaker-weight 1.0 \
  --original-topo false \
  --sort-strategy speech_time \
  --max-duration "$max_duration" \
  --max-train-batches "$max_train_batches" \
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
  --num-workers 6 \
  --use-fp16 true \
  --save-every-n 2000 \
  --average-period 200 \
  --tensorboard true \
  --seed 42
