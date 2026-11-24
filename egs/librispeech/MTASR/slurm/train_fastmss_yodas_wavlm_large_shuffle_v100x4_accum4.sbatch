#!/usr/bin/env bash
#SBATCH --job-name=mtasr-wavlm-tok-fmyod
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

export TORCH_HOME=${TORCH_HOME:-$recipe_dir/.cache/torch}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
library_path="$recipe_dir/.venv/lib/python3.11/site-packages/torch/lib:$recipe_dir/.venv/lib/python3.11/site-packages/k2/lib64"
if [[ -n "${CUDA_LIB_DIR:-}" ]]; then
  library_path="$library_path:$CUDA_LIB_DIR"
fi
export LD_LIBRARY_PATH="$library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exp_dir=${EXP_DIR:-exp/fastmss_yodas_en_wavlm_large_shuffle_v100x4_accum4}
start_batch=${START_BATCH:-0}
max_duration=${MAX_DURATION:-180}
yodas_shar_dir=${YODAS_SHAR_DIR:?Set YODAS_SHAR_DIR to the filtered English YODAS SHAR directory}
mkdir -p exp/slurm "$exp_dir" "$TORCH_HOME"

python wavlm_spashl_ctc/train.py \
  --world-size 4 \
  --master-port 15381 \
  --start-batch "$start_batch" \
  --exp-dir "$exp_dir" \
  --train-cut-manifests \
    data/manifests/fastmss_librispeech/shard1_cuts_groups_pause1.0.jsonl.gz \
    data/manifests/fastmss_librispeech/shard2_cuts_groups_pause1.0.jsonl.gz \
    data/manifests/fastmss_librispeech/shard3_cuts_groups_pause1.0.jsonl.gz \
    data/manifests/fastmss_librispeech/shard4_cuts_groups_pause1.0.jsonl.gz \
  --train-yodas-scheme-dirs \
    "$yodas_shar_dir" \
  --yodas-hypothesis doverlap \
  --yodas-group-max-pause 1.0 \
  --yodas-group-max-duration 60 \
  --yodas-min-speakers 2 \
  --yodas-max-speakers 8 \
  --yodas-mix-weight 0.5 \
  --bpe-model data/lang_bpe_5000_librispeech_yodas_en/bpe.model \
  --lang-dir data/lang_bpe_5000_librispeech_yodas_en \
  --use-large true \
  --speaker-branch false \
  --factorized-token-speaker-output false \
  --factorized-ctc-loss false \
  --downsample false \
  --use-layer-norm false \
  --max-num-spks 8 \
  --max-overlap 4 \
  --max-train-cut-duration 80 \
  --collar 32000 \
  --edge-collar 0 \
  --interval-model brownian_bridge \
  --beam-size 10 \
  --total-max-overlaps 6000 \
  --base-lr 0.00005 \
  --pct-start 0.15 \
  --total-steps 200000 \
  --freeze-iters 2000 \
  --freeze-lr 0.005 \
  --valid-interval 3000 \
  --original-topo false \
  --sort-strategy speech_time \
  --max-duration "$max_duration" \
  --accum-grad 4 \
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
