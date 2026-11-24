#!/usr/bin/env bash
#SBATCH --job-name=mtasr-parakeet-native-spashl
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
source activate.sh

export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

start_batch=${START_BATCH:-0}
max_duration=${MAX_DURATION:-80}
total_steps=${TOTAL_STEPS:-200000}
exp_dir=${EXP_DIR:-exp/parakeet_native_spashl}
mkdir -p exp/slurm "$exp_dir" "$HF_HOME"

python parakeet_native_spashl_ctc/train.py \
  --world-size 1 \
  --start-batch "$start_batch" \
  --exp-dir "$exp_dir" \
  --bpe-model data/lang_bpe_5000/bpe.model \
  --lang-dir data/lang_bpe_5000 \
  --parakeet-model-id nvidia/parakeet-tdt-0.6b-v3 \
  --factorized-token-speaker-output true \
  --speaker-branch true \
  --downsample false \
  --use-layer-norm false \
  --max-num-spks 4 \
  --max-overlap 3 \
  --speaker-weight 1.0 \
  --collar 64000 \
  --interval-model fixed \
  --beam-size 30 \
  --total-max-overlaps 6000 \
  --base-lr 0.00005 \
  --pct-start 0.15 \
  --total-steps "$total_steps" \
  --freeze-iters 2000 \
  --freeze-lr 0.005 \
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
  --num-workers 6 \
  --use-fp16 true \
  --save-every-n 2000 \
  --keep-last-k 5 \
  --average-period 200 \
  --tensorboard true \
  --seed 42
