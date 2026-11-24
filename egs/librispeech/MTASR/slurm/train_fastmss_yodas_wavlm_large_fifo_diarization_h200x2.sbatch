#!/usr/bin/env bash
#SBATCH --job-name=mtasr-wavlm-fifo-diar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:2
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
export LD_LIBRARY_PATH="$recipe_dir/.venv/lib/python3.11/site-packages/torch/lib:$recipe_dir/.venv/lib/python3.11/site-packages/k2/lib64:/expscratch/mwiesner/shuffle/.tools/cuda118/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

diarization_loss=${DIARIZATION_LOSS:?Set DIARIZATION_LOSS to bce or powerset}
case "$diarization_loss" in
  bce|powerset) ;;
  *) echo "DIARIZATION_LOSS must be bce or powerset" >&2; exit 2 ;;
esac

exp_dir=${EXP_DIR:-exp/fastmss_yodas_en_wavlm_large_fifo_${diarization_loss}_h200x2}
start_batch=${START_BATCH:-0}
max_duration=${MAX_DURATION:-1000}
world_size=${WORLD_SIZE:-2}
master_port=${MASTER_PORT:-15395}
total_steps=${TOTAL_STEPS:-200000}
freeze_iters=${FREEZE_ITERS:-2000}
skip_validation=${SKIP_VALIDATION:-false}
max_train_batches=${MAX_TRAIN_BATCHES:-0}
num_cuts_for_bins_estimate=${NUM_CUTS_FOR_BINS_ESTIMATE:-10000}
mkdir -p exp/slurm "$exp_dir" "$TORCH_HOME"

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
  --diarization-loss "$diarization_loss" \
  --speaker-branch false \
  --factorized-token-speaker-output false \
  --factorized-ctc-loss false \
  --use-large true \
  --downsample false \
  --use-layer-norm false \
  --max-num-spks 8 \
  --max-overlap 4 \
  --max-train-cut-duration 80 \
  --base-lr 0.00005 \
  --pct-start 0.15 \
  --total-steps "$total_steps" \
  --freeze-iters "$freeze_iters" \
  --freeze-lr 0.005 \
  --valid-interval 3000 \
  --skip-validation "$skip_validation" \
  --sort-strategy start_time \
  --max-duration "$max_duration" \
  --accum-grad 1 \
  --max-train-batches "$max_train_batches" \
  --num-buckets 100 \
  --num-cuts-for-bins-estimate "$num_cuts_for_bins_estimate" \
  --sampler-buffer-size 20000 \
  --num-workers 6 \
  --use-fp16 true \
  --save-every-n 2000 \
  --keep-last-k 5 \
  --average-period 200 \
  --tensorboard true \
  --seed 42
