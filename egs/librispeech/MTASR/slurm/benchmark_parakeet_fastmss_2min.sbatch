#!/usr/bin/env bash
#SBATCH --job-name=parakeet-2min-memory
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail

recipe_dir=${RECIPE_DIR:-$SLURM_SUBMIT_DIR}
cd "$recipe_dir"
source activate.sh
export LD_LIBRARY_PATH="$recipe_dir/.venv/lib/python3.11/site-packages/torch/lib:$recipe_dir/.venv/lib/python3.11/site-packages/k2/lib64:/expscratch/mwiesner/shuffle/.tools/cuda118/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_HOME=${HF_HOME:-/expscratch/mwiesner/shuffle/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

batch_size=${BATCH_SIZE:-1}
world_size=${WORLD_SIZE:-1}
max_train_batches=${MAX_TRAIN_BATCHES:-1}
accum_grad=${ACCUM_GRAD:-1}
train_manifest=${TRAIN_MANIFEST:-/expscratch/mwiesner/shuffle/FastMSS_LibriSpeech/librispeech/shard1/manifests/synth-librispeech-train-cuts.jsonl.gz}
profile_cuda_memory=${PROFILE_CUDA_MEMORY:-false}
max_duration=${MAX_DURATION:-100000}
exp_dir=${EXP_DIR:-exp/parakeet_hires_adapter_memory_b${batch_size}}
mkdir -p exp/slurm "$exp_dir"

python parakeet_native_spashl_ctc/train.py \
  --world-size "$world_size" \
  --exp-dir "$exp_dir" \
  --train-cut-manifests \
    "$train_manifest" \
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
  --max-train-cut-duration 130 \
  --speaker-weight 1.0 \
  --collar 32000 \
  --edge-collar 0 \
  --interval-model brownian_bridge \
  --beam-size 10 \
  --total-max-overlaps 6000 \
  --base-lr 0.00005 \
  --pct-start 0.15 \
  --total-steps 10 \
  --freeze-iters 0 \
  --original-topo false \
  --sort-strategy speech_time \
  --max-duration "$max_duration" \
  --max-cuts "$batch_size" \
  --num-buckets 10 \
  --num-cuts-for-bins-estimate 100 \
  --sampler-buffer-size 200 \
  --num-workers 0 \
  --use-fp16 true \
  --max-train-batches "$max_train_batches" \
  --accum-grad "$accum_grad" \
  --skip-validation true \
  --save-every-n 1000 \
  --average-period 200 \
  --tensorboard false \
  --profile-cuda-memory "$profile_cuda_memory" \
  --seed 42
