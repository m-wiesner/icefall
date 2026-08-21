#!/usr/bin/env bash
#SBATCH --job-name=mtasr-fastmss-brownian
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail

recipe_dir=${RECIPE_DIR:-$SLURM_SUBMIT_DIR}
cd "$recipe_dir"
source activate.sh

export TORCH_HOME=${TORCH_HOME:-/expscratch/mwiesner/shuffle/.cache/torch}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

start_batch=${START_BATCH:-0}
max_duration=${MAX_DURATION:-180}
max_train_cut_duration=${MAX_TRAIN_CUT_DURATION:-80}
total_steps=${TOTAL_STEPS:-200000}
exp_dir=${EXP_DIR:-exp/fastmss_wavlm_large_brownian_8spk}
mkdir -p exp/slurm "$exp_dir" "$TORCH_HOME"

python wavlm_spashl_ctc/train.py \
  --world-size 4 \
  --start-batch "$start_batch" \
  --exp-dir "$exp_dir" \
  --train-cut-manifests \
    data/manifests/fastmss_librispeech/shard1_cuts_groups_pause0.5.jsonl.gz \
    data/manifests/fastmss_librispeech/shard2_cuts_groups_pause0.5.jsonl.gz \
    data/manifests/fastmss_librispeech/shard3_cuts_groups_pause0.5.jsonl.gz \
    data/manifests/fastmss_librispeech/shard4_cuts_groups_pause0.5.jsonl.gz \
  --bpe-model data/lang_bpe_5000/bpe.model \
  --lang-dir data/lang_bpe_5000 \
  --use-large true \
  --factorized-token-speaker-output true \
  --downsample false \
  --use-layer-norm false \
  --max-num-spks 8 \
  --max-overlap 4 \
  --max-train-cut-duration "$max_train_cut_duration" \
  --collar 32000 \
  --edge-collar 0 \
  --interval-model brownian_bridge \
  --beam-size 10 \
  --total-max-overlaps 6000 \
  --base-lr 0.00005 \
  --pct-start 0.15 \
  --total-steps "$total_steps" \
  --freeze-iters 2000 \
  --freeze-lr 0.005 \
  --speaker-weight 1.0 \
  --original-topo false \
  --sort-strategy speech_time \
  --max-duration "$max_duration" \
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
