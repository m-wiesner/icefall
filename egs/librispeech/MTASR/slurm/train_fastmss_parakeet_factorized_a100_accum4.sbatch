#!/usr/bin/env bash
#SBATCH --job-name=mtasr-par-a100-factor
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
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
export LD_LIBRARY_PATH="$recipe_dir/.venv/lib/python3.11/site-packages/torch/lib:$recipe_dir/.venv/lib/python3.11/site-packages/k2/lib64:/expscratch/mwiesner/shuffle/.tools/cuda118/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exp_dir=${EXP_DIR:-exp/fastmss_parakeet_hires_local_factorized_pause1_a100_accum4}
start_batch=${START_BATCH:-0}
mkdir -p exp/slurm "$exp_dir"

python parakeet_native_spashl_ctc/train.py \
  --world-size 2 \
  --exp-dir "$exp_dir" \
  --start-batch "$start_batch" \
  --train-cut-manifests \
    data/manifests/fastmss_librispeech/shard1_cuts_groups_pause1.0.jsonl.gz \
    data/manifests/fastmss_librispeech/shard2_cuts_groups_pause1.0.jsonl.gz \
    data/manifests/fastmss_librispeech/shard3_cuts_groups_pause1.0.jsonl.gz \
    data/manifests/fastmss_librispeech/shard4_cuts_groups_pause1.0.jsonl.gz \
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
  --factorized-ctc-loss true \
  --speaker-branch true \
  --downsample false \
  --use-layer-norm false \
  --max-num-spks 8 \
  --max-overlap 4 \
  --max-train-cut-duration 100 \
  --speaker-weight 1.0 \
  --collar 32000 \
  --edge-collar 0 \
  --interval-model brownian_bridge \
  --beam-size 10 \
  --total-max-overlaps 6000 \
  --base-lr 0.00005 \
  --pct-start 0.15 \
  --total-steps 200000 \
  --freeze-iters 5000 \
  --freeze-lr 0.00002 \
  --original-topo false \
  --sort-strategy speech_time \
  --max-duration 2700 \
  --max-cuts "${MAX_CUTS:-2}" \
  --accum-grad 4 \
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
