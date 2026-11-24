#!/usr/bin/env bash
#SBATCH --job-name=mtasr-fastmss-parakeet
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

export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

start_batch=${START_BATCH:-0}
max_duration=${MAX_DURATION:-2700}
max_train_cut_duration=${MAX_TRAIN_CUT_DURATION:-130}
total_steps=${TOTAL_STEPS:-200000}
exp_dir=${EXP_DIR:-exp/fastmss_parakeet_hires_local_adapter_spashl_8spk_fast_sampler}
mkdir -p exp/slurm "$exp_dir"

python parakeet_native_spashl_ctc/train.py \
  --world-size 4 \
  --start-batch "$start_batch" \
  --exp-dir "$exp_dir" \
  --train-cut-manifests \
    /expscratch/mwiesner/shuffle/FastMSS_LibriSpeech/librispeech/shard1/manifests/synth-librispeech-train-cuts.jsonl.gz \
    /expscratch/mwiesner/shuffle/FastMSS_LibriSpeech/librispeech/shard2/manifests/synth-librispeech-train-cuts.jsonl.gz \
    /expscratch/mwiesner/shuffle/FastMSS_LibriSpeech/librispeech/shard3/manifests/synth-librispeech-train-cuts.jsonl.gz \
    /expscratch/mwiesner/shuffle/FastMSS_LibriSpeech/librispeech/shard4/manifests/synth-librispeech-train-cuts.jsonl.gz \
  --bpe-model data/lang_bpe_5000/bpe.model \
  --lang-dir data/lang_bpe_5000 \
  --parakeet-model-id nvidia/parakeet-tdt-0.6b-v3 \
  --adapter-dim 256 \
  --adapter-num-heads 4 \
  --adapter-ff-dim 1024 \
  --adapter-layer-indices 7,15,23 \
  --adapter-local-attention-window 257 \
  --adapter-cross-attention-window 65 \
  --adapter-attention-block-size 256 \
  --adapter-dropout 0.1 \
  --factorized-token-speaker-output true \
  --speaker-branch true \
  --downsample false \
  --use-layer-norm false \
  --max-num-spks 8 \
  --max-overlap 4 \
  --max-train-cut-duration "$max_train_cut_duration" \
  --speaker-weight 1.0 \
  --collar 32000 \
  --edge-collar 0 \
  --interval-model brownian_bridge \
  --beam-size 10 \
  --total-max-overlaps 6000 \
  --base-lr 0.00005 \
  --pct-start 0.15 \
  --total-steps "$total_steps" \
  --freeze-iters 5000 \
  --freeze-lr 0.00002 \
  --original-topo false \
  --sort-strategy speech_time \
  --max-duration "$max_duration" \
  --max-cuts 2 \
  --num-buckets 10 \
  --num-cuts-for-bins-estimate 100 \
  --sampler-buffer-size 200 \
  --num-workers 6 \
  --use-fp16 true \
  --save-every-n 2000 \
  --keep-last-k 5 \
  --average-period 200 \
  --tensorboard true \
  --seed 42
