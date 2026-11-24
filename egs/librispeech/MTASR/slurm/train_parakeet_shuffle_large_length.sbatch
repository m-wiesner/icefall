#!/usr/bin/env bash
#SBATCH --job-name=mtasr-parakeet
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
export HF_HOME=${HF_HOME:-$recipe_dir/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

start_batch=${START_BATCH:-0}
world_size=${WORLD_SIZE:-1}
max_duration=${MAX_DURATION:-80}
total_steps=${TOTAL_STEPS:-200000}
beam_size=${BEAM_SIZE:-30}
max_train_batches=${MAX_TRAIN_BATCHES:-0}
validation_batchnorm_train_mode=${VALIDATION_BATCHNORM_TRAIN_MODE:-false}
validation_only=${VALIDATION_ONLY:-false}
parakeet_recipe=${PARAKEET_RECIPE:-parakeet_spashl_ctc}
speaker_branch=${SPEAKER_BRANCH:-true}
speaker_layer=${SPEAKER_LAYER:-12}
convolution_norm=${PARAKEET_CONVOLUTION_NORM:-masked_batch_norm}
finite_diagnostics=${PARAKEET_FINITE_DIAGNOSTICS:-false}
high_resolution_fusion=${HIGH_RESOLUTION_FUSION:-false}
recovery_adapter_blocks=${RECOVERY_ADAPTER_BLOCKS:-2}
exp_dir=${EXP_DIR:-exp/parakeet_spashl_learned_phase}
mkdir -p exp/slurm "$exp_dir"

python "$parakeet_recipe/train.py" \
  --world-size "$world_size" \
  --start-batch "$start_batch" \
  --max-train-batches "$max_train_batches" \
  --validation-batchnorm-train-mode "$validation_batchnorm_train_mode" \
  --validation-only "$validation_only" \
  --check-batchnorm-finite-interval 100 \
  --exp-dir "$exp_dir" \
  --bpe-model data/lang_bpe_5000/bpe.model \
  --lang-dir data/lang_bpe_5000 \
  --parakeet-model-id nvidia/parakeet-tdt-0.6b-v3 \
  --encoder-upsample-factor 4 \
  --encoder-upsample-mode linear \
  --high-resolution-fusion "$high_resolution_fusion" \
  --recovery-adapter-blocks "$recovery_adapter_blocks" \
  --speaker-layer "$speaker_layer" \
  --encoder-layerdrop 0.0 \
  --parakeet-convolution-norm "$convolution_norm" \
  --parakeet-finite-diagnostics "$finite_diagnostics" \
  --downsample false \
  --use-layer-norm false \
  --max-num-spks 4 \
  --max-overlap 3 \
  --factorized-token-speaker-output true \
  --speaker-branch "$speaker_branch" \
  --speaker-weight 1.0 \
  --collar 64000 \
  --interval-model fixed \
  --beam-size "$beam_size" \
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
