#!/usr/bin/env bash
#SBATCH --job-name=mtasr-wl-tokseg-v8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --output=exp/slurm/%x-%j.out
#SBATCH --error=exp/slurm/%x-%j.err

set -euo pipefail

recipe_dir=${RECIPE_DIR:-$SLURM_SUBMIT_DIR}
cd "$recipe_dir"
source activate.sh

export TORCH_HOME=${TORCH_HOME:-/expscratch/mwiesner/shuffle/.cache/torch}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$recipe_dir/.venv/lib/python3.11/site-packages/torch/lib:$recipe_dir/.venv/lib/python3.11/site-packages/k2/lib64:/expscratch/mwiesner/shuffle/.tools/cuda118/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exp_dir=${EXP_DIR:-exp/fastmss_yodas_en_wavlm_large_shuffle_segment_factor_softmask_v100x8_accum3}
start_batch=${START_BATCH:-0}
max_duration=${MAX_DURATION:-120}
world_size=${WORLD_SIZE:-8}
master_port=${MASTER_PORT:-15399}
total_steps=${TOTAL_STEPS:-200000}
freeze_iters=${FREEZE_ITERS:-2000}
accum_grad=${ACCUM_GRAD:-3}
max_train_batches=${MAX_TRAIN_BATCHES:-0}
skip_validation=${SKIP_VALIDATION:-false}
mkdir -p exp/slurm "$exp_dir" "$TORCH_HOME"

# Match the previous effective batch duration:
#   old: 4 GPUs x 180 s/GPU x 4 accumulation = 2880 s/update
#   new: 8 GPUs x 120 s/GPU x 3 accumulation = 2880 s/update
python wavlm_spashl_ctc/train.py \
  --world-size "$world_size" \
  --master-port "$master_port" \
  --start-batch "$start_batch" \
  --exp-dir "$exp_dir" \
  --train-cut-manifests \
    data/manifests/fastmss_librispeech/shard1_cuts_groups_pause1.0.jsonl.gz \
    data/manifests/fastmss_librispeech/shard2_cuts_groups_pause1.0.jsonl.gz \
    data/manifests/fastmss_librispeech/shard3_cuts_groups_pause1.0.jsonl.gz \
    data/manifests/fastmss_librispeech/shard4_cuts_groups_pause1.0.jsonl.gz \
  --train-yodas-scheme-dirs \
    /expscratch/mwiesner/shuffle/yodas_shar_filtered/en_judge8_der0069_doverlap \
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
  --factorized-token-speaker-output true \
  --factorized-ctc-loss true \
  --token-only-segment-factor true \
  --segment-factor-collar 48000 \
  --segment-factor-decay-scale 16000 \
  --segment-factor-floor-prob 0.0001 \
  --determinize-token-only-graph false \
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
  --total-steps "$total_steps" \
  --freeze-iters "$freeze_iters" \
  --freeze-lr 0.005 \
  --valid-interval 3000 \
  --original-topo false \
  --sort-strategy speech_time \
  --max-duration "$max_duration" \
  --accum-grad "$accum_grad" \
  --max-train-batches "$max_train_batches" \
  --skip-validation "$skip_validation" \
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
